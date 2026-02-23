"""
Optimizer for VP Confirmed Breakout strategy.
Strategy: precompute VP levels once per va_percent, then fast-sweep trading params.
"""
import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import sys
import warnings
from collections import defaultdict
import time
import pickle

warnings.filterwarnings('ignore')
load_dotenv()

# =============================================================================
# FIXED CONFIG
# =============================================================================
START_DATE_STR = "2025-01-01 00:00:00"
INITIAL_CAPITAL = 1000
RISK_PERCENT = 0.01

ASSET = {
    'symbol': 'XAUUSD',
    'candle_table': 'candles_mt5_xauusd_1m',
    'tick_table': 'market_ticks_xauusd',
    'tick_size': 0.01,
}

SESSIONS_CONFIG = {
    'TOKYO':  {'vp_start': 0,    'vp_end': 4,    'trade_start': 0,    'trade_end': 4},
    'LONDON': {'vp_start': 8,    'vp_end': 14.5, 'trade_start': 9,    'trade_end': 14},
    'NY':     {'vp_start': 14.5, 'vp_end': 21.5, 'trade_start': 15,   'trade_end': 21},
}


# =============================================================================
# VP classes (same as original)
# =============================================================================
class IncrementalVolumeProfile:
    def __init__(self, tick_size=0.01, va_percent=0.70):
        self.tick_size = tick_size
        self.va_percent = va_percent
        self.reset()

    def reset(self):
        self.profile = defaultdict(float)
        self.total_volume = 0.0
        self._cache_valid = False
        self._cached_poc = None
        self._cached_vah = None
        self._cached_val = None
        self._cached_poc_strength = None

    def add_ticks(self, prices, volumes):
        if len(prices) == 0:
            return
        price_bins = np.round(prices / self.tick_size) * self.tick_size
        unique_bins, indices = np.unique(price_bins, return_inverse=True)
        bin_volumes = np.bincount(indices, weights=volumes)
        for price_bin, vol in zip(unique_bins, bin_volumes):
            self.profile[price_bin] += vol
            self.total_volume += vol
        self._cache_valid = False

    def _compute_value_area(self):
        if not self.profile:
            self._cached_poc = self._cached_vah = self._cached_val = None
            self._cached_poc_strength = None
            self._cache_valid = True
            return
        sorted_bins = sorted(self.profile.keys())
        volumes = np.array([self.profile[b] for b in sorted_bins])
        poc_idx = np.argmax(volumes)
        self._cached_poc = sorted_bins[poc_idx]
        poc_volume = volumes[poc_idx]
        avg_volume = np.mean(volumes)
        self._cached_poc_strength = poc_volume / avg_volume if avg_volume > 0 else 0
        target_volume = self.total_volume * self.va_percent
        current_volume = volumes[poc_idx]
        up_idx, down_idx = poc_idx + 1, poc_idx - 1
        while current_volume < target_volume:
            vol_up = volumes[up_idx] if up_idx < len(volumes) else 0
            vol_down = volumes[down_idx] if down_idx >= 0 else 0
            if vol_up == 0 and vol_down == 0:
                break
            if vol_up >= vol_down:
                current_volume += vol_up
                up_idx += 1
            else:
                current_volume += vol_down
                down_idx -= 1
        self._cached_vah = sorted_bins[min(up_idx - 1, len(sorted_bins) - 1)]
        self._cached_val = sorted_bins[max(down_idx + 1, 0)]
        self._cache_valid = True

    def get_levels(self):
        if not self._cache_valid:
            self._compute_value_area()
        return self._cached_poc, self._cached_vah, self._cached_val

    def get_poc_strength(self):
        if not self._cache_valid:
            self._compute_value_area()
        return self._cached_poc_strength

    def snapshot(self):
        poc, vah, val = self.get_levels()
        return {'poc': poc, 'vah': vah, 'val': val}


class StructuralLevelsTracker:
    def __init__(self, tick_size=0.01, va_percent=0.70):
        self.tick_size = tick_size
        self.va_percent = va_percent
        self._daily_vp = IncrementalVolumeProfile(tick_size, va_percent)
        self._weekly_vp = IncrementalVolumeProfile(tick_size, va_percent)
        self.prev_day = {'poc': None, 'vah': None, 'val': None}
        self.prev_week = {'poc': None, 'vah': None, 'val': None}
        self._current_day = None
        self._current_week_start = None

    def update(self, dt, prices, volumes):
        day = dt.date()
        week_start = (dt - timedelta(days=dt.weekday())).date()
        if self._current_day is not None and day != self._current_day:
            snap = self._daily_vp.snapshot()
            if snap['poc'] is not None:
                self.prev_day = snap
            self._daily_vp.reset()
        if self._current_week_start is not None and week_start != self._current_week_start:
            snap = self._weekly_vp.snapshot()
            if snap['poc'] is not None:
                self.prev_week = snap
            self._weekly_vp.reset()
        self._current_day = day
        self._current_week_start = week_start
        if len(prices) > 0:
            self._daily_vp.add_ticks(prices, volumes)
            self._weekly_vp.add_ticks(prices, volumes)


# =============================================================================
# HELPERS
# =============================================================================
def get_session(dt):
    h = dt.hour + dt.minute / 60.0
    for sess_name, cfg in SESSIONS_CONFIG.items():
        if cfg['vp_start'] <= h < cfg['vp_end']:
            return sess_name
    return "AUTRE"

def is_session_start(dt):
    current_time = dt.hour + dt.minute / 60.0
    for sess_name, cfg in SESSIONS_CONFIG.items():
        if current_time == cfg['vp_start']:
            return sess_name
    return None

def can_trade_now(dt, session_name):
    if session_name not in SESSIONS_CONFIG:
        return False
    cfg = SESSIONS_CONFIG[session_name]
    current_time = dt.hour + dt.minute / 60.0
    return cfg['trade_start'] <= current_time < cfg['trade_end']

def get_session_start_time(session_name, reference_dt):
    if session_name not in SESSIONS_CONFIG:
        return None
    cfg = SESSIONS_CONFIG[session_name]
    start_hour = cfg['vp_start']
    hour = int(start_hour)
    minute = int((start_hour % 1) * 60)
    return reference_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


# =============================================================================
# STEP 1: Load data
# =============================================================================
def load_data():
    conn = psycopg2.connect(
        host=os.getenv('PG_HOST'), port=os.getenv('PG_PORT'),
        database=os.getenv('PG_DB'), user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD')
    )
    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = requested_start - timedelta(days=14)
    ts_start = int(data_start.timestamp() * 1000)

    print("[DATA] Loading candles...")
    query_candles = f"SELECT ts, open, high, low, close FROM {ASSET['candle_table']} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df_candles = pd.read_sql(query_candles, conn)
    df_candles['dt'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)
    print(f"   {len(df_candles):,} candles")

    print("[DATA] Loading ticks...")
    t_start = data_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = df_candles['dt'].max().strftime("%Y-%m-%d %H:%M:%S")
    query_ticks = f"""SELECT time, last as price, volume FROM {ASSET['tick_table']}
        WHERE time >= '{t_start}' AND time <= '{t_end}'
        ORDER BY time ASC"""
    df_ticks = pd.read_sql(query_ticks, conn)
    df_ticks['time'] = pd.to_datetime(df_ticks['time'], utc=True)
    df_ticks['minute'] = df_ticks['time'].dt.floor('T')
    print(f"   {len(df_ticks):,} ticks")

    ticks_by_minute = df_ticks.groupby('minute').apply(
        lambda g: (g['price'].values, g['volume'].values, g['time'].values)
    ).to_dict()

    conn.close()
    return df_candles, ticks_by_minute, requested_start


# =============================================================================
# STEP 2: Precompute candle-level data with VP levels for a given va_percent
# This is the expensive step - we do it once per va_percent
# =============================================================================
def precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70):
    """
    For each candle, compute:
    - session VP levels (poc, vah, val, poc_strength)
    - structural levels (prev day/week)
    - session info, trade window, etc.
    Returns a list of dicts, one per candle (post-warmup only).
    """
    tick_size = ASSET['tick_size']
    vp = IncrementalVolumeProfile(tick_size=tick_size, va_percent=va_percent)
    structural = StructuralLevelsTracker(tick_size=tick_size, va_percent=va_percent)
    session_start_dt = None

    candle_data = []

    for row in df_candles.itertuples():
        current_minute = row.dt.floor('T')

        # Structural (always)
        if current_minute in ticks_by_minute:
            prices, volumes, timestamps = ticks_by_minute[current_minute]
            structural.update(row.dt, prices, volumes)

        if row.dt < requested_start:
            continue

        curr_sess = get_session(row.dt)
        sess_start = is_session_start(row.dt)
        if sess_start:
            vp.reset()
            session_start_dt = get_session_start_time(sess_start, row.dt)

        # Session VP ticks
        in_session = curr_sess in SESSIONS_CONFIG
        if in_session and current_minute in ticks_by_minute:
            prices, volumes, timestamps = ticks_by_minute[current_minute]
            if session_start_dt is not None:
                session_start_np = np.datetime64(session_start_dt)
                mask = timestamps >= session_start_np
                if mask.any():
                    vp.add_ticks(prices[mask], volumes[mask])
            else:
                vp.add_ticks(prices, volumes)

        poc, vah, val = vp.get_levels()
        poc_strength = vp.get_poc_strength()

        can_trade = can_trade_now(row.dt, curr_sess) if in_session else False

        candle_data.append({
            'dt': row.dt,
            'open': row.open,
            'high': row.high,
            'low': row.low,
            'close': row.close,
            'session': curr_sess,
            'session_start': sess_start,
            'in_session': in_session,
            'can_trade': can_trade,
            'hour': row.dt.hour,
            'day_of_week': row.dt.weekday(),
            'poc': poc,
            'vah': vah,
            'val': val,
            'poc_strength': poc_strength,
            # Structural levels snapshot
            'pd_vah': structural.prev_day['vah'],
            'pd_val': structural.prev_day['val'],
            'pd_poc': structural.prev_day['poc'],
            'pw_vah': structural.prev_week['vah'],
            'pw_val': structural.prev_week['val'],
            'pw_poc': structural.prev_week['poc'],
        })

    return candle_data


# =============================================================================
# STEP 3: Fast backtest on precomputed data
# =============================================================================
def fast_backtest(candle_data, params):
    """
    Run backtest using precomputed candle data (with VP levels baked in).
    Only iterates through candle_data without any VP computation.
    """
    wait_candles = params.get('wait_candles', 4)
    min_rr = params.get('min_rr', 2.0)
    tp1_rr = params.get('tp1_rr', 1.3)
    tp1_split = params.get('tp1_split', 0.5)
    tp2_split = 1.0 - tp1_split
    sl_offset = params.get('sl_offset', 0.50)
    min_poc_strength = params.get('min_poc_strength', 2.5)
    use_trailing = params.get('use_trailing', True)
    use_prev_day = params.get('use_prev_day', True)
    use_prev_week = params.get('use_prev_week', True)
    sessions_enabled = params.get('sessions', {'TOKYO': True, 'LONDON': True, 'NY': True})
    allowed_days = set(params.get('allowed_days', [0, 1, 2, 3, 4]))
    excluded_hours = set(params.get('excluded_hours', []))
    allow_long = params.get('allow_long', True)
    allow_short = params.get('allow_short', True)
    use_cooldown = params.get('use_cooldown', False)
    cooldown_minutes = params.get('cooldown_minutes', 60)
    exclude_vah_target = params.get('exclude_vah_target', False)
    max_rr_cap = params.get('max_rr_cap', 30)
    only_structural_tp = params.get('only_structural_tp', False)  # skip if no structural target

    state = "INSIDE"
    active_trade = None
    breakout_direction = None
    candles_since_breakout = 0
    wait_highs = []
    wait_lows = []
    last_loss_time = None
    last_loss_direction = None

    current_capital = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    max_dd_percent = 0.0
    max_dd_amount = 0.0

    total_trades = 0
    wins = 0
    be_trades = 0
    losses = 0
    total_pnl_r = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    # Monthly tracking for consistency
    monthly_pnl = defaultdict(float)

    for c in candle_data:
        dt = c['dt']
        close = c['close']
        high = c['high']
        low = c['low']
        curr_sess = c['session']
        poc = c['poc']
        vah = c['vah']
        val = c['val']
        poc_strength = c['poc_strength']

        # Session reset
        if c['session_start'] and sessions_enabled.get(c['session_start'], False):
            state = "INSIDE"
            breakout_direction = None
            candles_since_breakout = 0
            wait_highs = []
            wait_lows = []

        # Manage active trade
        if active_trade:
            res = None
            if active_trade['type'] == 'LONG':
                if low <= active_trade['sl']:
                    res = "LOSS"
                else:
                    if use_trailing and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] + (active_trade['risk'] * tp1_rr)
                        if high >= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = tp1_rr * tp1_split
                    if high >= active_trade['tp']:
                        res = "WIN"
            elif active_trade['type'] == 'SHORT':
                if high >= active_trade['sl']:
                    res = "LOSS"
                else:
                    if use_trailing and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] - (active_trade['risk'] * tp1_rr)
                        if low <= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = tp1_rr * tp1_split
                    if low <= active_trade['tp']:
                        res = "WIN"

            if res:
                risk_amount = current_capital * RISK_PERCENT
                trade_rr = active_trade['rr']
                partial_pnl_r = active_trade.get('partial_pnl_r', 0)
                month_key = dt.strftime('%Y-%m')
                if res == "WIN":
                    if use_trailing and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r + (trade_rr * tp2_split)
                    else:
                        pnl_r = trade_rr
                    pnl = risk_amount * pnl_r
                    wins += 1
                else:
                    if use_trailing and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r
                        pnl = risk_amount * pnl_r
                        be_trades += 1
                    else:
                        pnl = -risk_amount
                        pnl_r = -1.0
                        losses += 1
                        if use_cooldown:
                            last_loss_time = dt
                            last_loss_direction = active_trade['type']

                current_capital += pnl
                total_trades += 1
                total_pnl_r += pnl_r
                monthly_pnl[month_key] += pnl_r
                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)

                if current_capital > high_water_mark:
                    high_water_mark = current_capital
                dd_pct = ((high_water_mark - current_capital) / high_water_mark * 100) if high_water_mark > 0 else 0
                if dd_pct > max_dd_percent:
                    max_dd_percent = dd_pct
                dd_amt = high_water_mark - current_capital
                if dd_amt > max_dd_amount:
                    max_dd_amount = dd_amt

                active_trade = None
            else:
                continue

        # State machine
        if not sessions_enabled.get(curr_sess, False):
            state = "INSIDE"
            breakout_direction = None
            candles_since_breakout = 0
            continue

        if poc is None:
            continue

        if state == "INSIDE":
            if close > vah:
                state = "WAITING"
                breakout_direction = "UP"
                candles_since_breakout = 1
                wait_highs = [high]
                wait_lows = [low]
            elif close < val:
                state = "WAITING"
                breakout_direction = "DOWN"
                candles_since_breakout = 1
                wait_highs = [high]
                wait_lows = [low]

        elif state == "WAITING":
            candles_since_breakout += 1
            wait_highs.append(high)
            wait_lows.append(low)

            if breakout_direction == "UP" and close <= vah:
                state = "INSIDE"
                breakout_direction = None
                candles_since_breakout = 0
                continue
            if breakout_direction == "DOWN" and close >= val:
                state = "INSIDE"
                breakout_direction = None
                candles_since_breakout = 0
                continue

            if candles_since_breakout >= wait_candles:
                direction = 'LONG' if breakout_direction == 'UP' else 'SHORT'

                poc_ok = poc_strength is not None and poc_strength >= min_poc_strength
                can_dir = (direction == 'LONG' and allow_long) or (direction == 'SHORT' and allow_short)
                can_time = c['can_trade']
                day_ok = c['day_of_week'] in allowed_days
                hour_ok = c['hour'] not in excluded_hours

                cooldown_ok = True
                if use_cooldown and last_loss_time is not None:
                    if last_loss_direction == direction:
                        mins = (dt - last_loss_time).total_seconds() / 60.0
                        if mins < cooldown_minutes:
                            cooldown_ok = False

                if poc_ok and can_dir and can_time and day_ok and hour_ok and cooldown_ok:
                    if direction == 'LONG':
                        swing_low = min(wait_lows)
                        sl = swing_low - sl_offset
                        risk = close - sl
                    else:
                        swing_high = max(wait_highs)
                        sl = swing_high + sl_offset
                        risk = sl - close

                    if risk > 0:
                        # Build targets
                        candidates = []
                        if use_prev_day:
                            if c['pd_vah'] is not None and not (exclude_vah_target and True):
                                candidates.append(('PD_VAH', c['pd_vah']))
                            if c['pd_val'] is not None:
                                candidates.append(('PD_VAL', c['pd_val']))
                            if c['pd_poc'] is not None:
                                candidates.append(('PD_POC', c['pd_poc']))
                        if use_prev_week:
                            if c['pw_vah'] is not None and not (exclude_vah_target and True):
                                candidates.append(('PW_VAH', c['pw_vah']))
                            if c['pw_val'] is not None:
                                candidates.append(('PW_VAL', c['pw_val']))
                            if c['pw_poc'] is not None:
                                candidates.append(('PW_POC', c['pw_poc']))

                        if exclude_vah_target:
                            candidates = [(l, p) for l, p in candidates if 'VAH' not in l]

                        if direction == 'LONG':
                            targets = [(l, p) for l, p in candidates if p > close]
                            targets.sort(key=lambda x: x[1])
                        else:
                            targets = [(l, p) for l, p in candidates if p < close]
                            targets.sort(key=lambda x: -x[1])

                        tp = None
                        for label, price in targets:
                            if direction == 'LONG':
                                rr = (price - close) / risk
                            else:
                                rr = (close - price) / risk
                            if rr >= min_rr and rr <= max_rr_cap:
                                tp = price
                                break

                        if tp is None:
                            if only_structural_tp:
                                # Skip - no structural target found
                                state = "INSIDE"
                                breakout_direction = None
                                candles_since_breakout = 0
                                continue
                            if direction == 'LONG':
                                tp = close + (risk * min_rr)
                            else:
                                tp = close - (risk * min_rr)
                            actual_rr = min_rr
                        else:
                            if direction == 'LONG':
                                actual_rr = (tp - close) / risk
                            else:
                                actual_rr = (close - tp) / risk

                        active_trade = {
                            'type': direction, 'entry': close, 'sl': sl, 'risk': risk,
                            'tp': tp, 'rr': actual_rr,
                            'partial_closed': False, 'partial_pnl_r': 0,
                        }

                state = "INSIDE"
                breakout_direction = None
                candles_since_breakout = 0

    # Results
    win_rate = (wins + be_trades) / total_trades * 100 if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_pnl = current_capital - INITIAL_CAPITAL
    avg_pnl_r = total_pnl_r / total_trades if total_trades > 0 else 0
    pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100

    # Monthly consistency
    positive_months = sum(1 for v in monthly_pnl.values() if v > 0)
    total_months = len(monthly_pnl)
    monthly_consistency = (positive_months / total_months * 100) if total_months > 0 else 0

    return {
        'total_trades': total_trades,
        'wins': wins,
        'be': be_trades,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl_r': total_pnl_r,
        'pnl_pct': pnl_pct,
        'profit_factor': profit_factor,
        'max_dd_pct': max_dd_percent,
        'avg_pnl_r': avg_pnl_r,
        'capital_final': current_capital,
        'monthly_consistency': monthly_consistency,
        'positive_months': positive_months,
        'total_months': total_months,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 110)
    print("OPTIMIZER - VP Confirmed Breakout (fast sweep)")
    print("=" * 110)

    # Load data
    df_candles, ticks_by_minute, requested_start = load_data()

    # Precompute candle data for va_percent=0.70 (default)
    print("\n[PRECOMPUTE] Building VP levels for VA=70%...")
    t0 = time.time()
    candle_data_70 = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70)
    print(f"   {len(candle_data_70):,} candles precomputed in {time.time()-t0:.1f}s")

    # Also precompute for other VA percents we want to test
    print("[PRECOMPUTE] Building VP levels for VA=65%...")
    t0 = time.time()
    candle_data_65 = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.65)
    print(f"   Done in {time.time()-t0:.1f}s")

    print("[PRECOMPUTE] Building VP levels for VA=75%...")
    t0 = time.time()
    candle_data_75 = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.75)
    print(f"   Done in {time.time()-t0:.1f}s")

    print("[PRECOMPUTE] Building VP levels for VA=60%...")
    t0 = time.time()
    candle_data_60 = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.60)
    print(f"   Done in {time.time()-t0:.1f}s")

    va_data = {
        0.60: candle_data_60,
        0.65: candle_data_65,
        0.70: candle_data_70,
        0.75: candle_data_75,
    }

    # =========================================================================
    # Build parameter grid
    # =========================================================================
    param_grid = []

    # --- PHASE 1: Individual parameter sweeps (on VA=70%) ---
    base = {
        'va_percent': 0.70, 'wait_candles': 4, 'min_rr': 2.0,
        'tp1_rr': 1.3, 'tp1_split': 0.5, 'sl_offset': 0.50,
        'min_poc_strength': 2.5, 'use_trailing': True,
        'use_prev_day': True, 'use_prev_week': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
        'allowed_days': [0, 1, 2, 3, 4],
        'excluded_hours': [],
    }

    param_grid.append(("BASELINE", {**base}))

    # Sessions
    param_grid.append(("TKY+LDN", {**base, 'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False}}))
    param_grid.append(("TKY_only", {**base, 'sessions': {'TOKYO': True, 'LONDON': False, 'NY': False}}))
    param_grid.append(("LDN_only", {**base, 'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False}}))
    param_grid.append(("NY_only", {**base, 'sessions': {'TOKYO': False, 'LONDON': False, 'NY': True}}))

    # Days
    param_grid.append(("Mon-Thu", {**base, 'allowed_days': [0, 1, 2, 3]}))
    param_grid.append(("Tue-Thu", {**base, 'allowed_days': [1, 2, 3]}))
    param_grid.append(("Mon-Wed", {**base, 'allowed_days': [0, 1, 2]}))

    # Hours exclusion
    param_grid.append(("NoBadH", {**base, 'excluded_hours': [0, 10, 17, 19, 20]}))
    param_grid.append(("NoBadH2", {**base, 'excluded_hours': [0, 10, 15, 17, 19, 20]}))

    # Wait candles
    for w in [2, 3, 5, 6, 8, 10]:
        param_grid.append((f"Wait{w}", {**base, 'wait_candles': w}))

    # Min RR
    for rr in [1.5, 2.5, 3.0, 4.0, 5.0]:
        param_grid.append((f"RR{rr}", {**base, 'min_rr': rr}))

    # SL offset
    for sl in [0.0, 0.25, 1.0, 1.5, 2.0, 3.0]:
        param_grid.append((f"SL{sl}", {**base, 'sl_offset': sl}))

    # TP1 RR
    for tp1 in [0.5, 0.8, 1.0, 1.5, 2.0]:
        param_grid.append((f"TP1_{tp1}R", {**base, 'tp1_rr': tp1}))

    # TP1 split
    for split in [0.3, 0.4, 0.6, 0.7]:
        param_grid.append((f"Split{int(split*100)}", {**base, 'tp1_split': split}))

    # No trailing
    param_grid.append(("NoTrailing", {**base, 'use_trailing': False}))

    # POC strength
    for poc in [1.5, 2.0, 3.0, 3.5, 4.0, 5.0]:
        param_grid.append((f"POC{poc}", {**base, 'min_poc_strength': poc}))

    # Exclude VAH targets
    param_grid.append(("NoVAH_TP", {**base, 'exclude_vah_target': True}))

    # Only structural TP (no fallback)
    param_grid.append(("OnlyStructTP", {**base, 'only_structural_tp': True}))
    param_grid.append(("OnlyStructTP_NoVAH", {**base, 'only_structural_tp': True, 'exclude_vah_target': True}))

    # Level sources
    param_grid.append(("OnlyPrevDay", {**base, 'use_prev_day': True, 'use_prev_week': False}))
    param_grid.append(("OnlyPrevWeek", {**base, 'use_prev_day': False, 'use_prev_week': True}))

    # Cooldown
    param_grid.append(("Cooldown60", {**base, 'use_cooldown': True, 'cooldown_minutes': 60}))
    param_grid.append(("Cooldown120", {**base, 'use_cooldown': True, 'cooldown_minutes': 120}))

    # Long only / Short only
    param_grid.append(("LongOnly", {**base, 'allow_long': True, 'allow_short': False}))
    param_grid.append(("ShortOnly", {**base, 'allow_long': False, 'allow_short': True}))

    # VA percent (use different precomputed data)
    param_grid.append(("VA60", {**base, 'va_percent': 0.60}))
    param_grid.append(("VA65", {**base, 'va_percent': 0.65}))
    param_grid.append(("VA75", {**base, 'va_percent': 0.75}))

    # =========================================================================
    # Run Phase 1
    # =========================================================================
    print(f"\n[PHASE 1] Running {len(param_grid)} individual parameter tests...")
    print(f"{'#':>3} | {'NAME':<25} | {'TRADES':>6} | {'WR%':>6} | {'PnL R':>8} | {'PnL%':>8} | {'PF':>6} | {'MaxDD%':>7} | {'AvgR':>6} | {'Consist':>6} | {'TIME':>6}")
    print("-" * 120)

    all_results = []
    for i, (name, params) in enumerate(param_grid):
        va = params.get('va_percent', 0.70)
        data = va_data.get(va, candle_data_70)
        t0 = time.time()
        r = fast_backtest(data, params)
        elapsed = time.time() - t0

        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(f"{i+1:>3} | {name:<25} | {r['total_trades']:>6} | {r['win_rate']:>5.1f}% | {r['total_pnl_r']:>+7.1f}R | {r['pnl_pct']:>+7.1f}% | {pf_str:>6} | {r['max_dd_pct']:>6.1f}% | {r['avg_pnl_r']:>+5.2f} | {r['monthly_consistency']:>5.0f}% | {elapsed:>5.1f}s")
        all_results.append((name, params, r))

    # =========================================================================
    # TOP 10 Phase 1
    # =========================================================================
    print("\n" + "=" * 110)
    print("PHASE 1 - TOP 10 BY PnL R (min 30 trades)")
    print("=" * 110)
    valid = [(n, p, r) for n, p, r in all_results if r['total_trades'] >= 30]
    valid.sort(key=lambda x: x[2]['total_pnl_r'], reverse=True)
    for i, (name, params, r) in enumerate(valid[:10]):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(f"  {i+1:>2}. {name:<25} | {r['total_trades']:>5} tr | WR:{r['win_rate']:>5.1f}% | PnL:{r['total_pnl_r']:>+7.1f}R ({r['pnl_pct']:>+6.1f}%) | PF:{pf_str:>5} | DD:{r['max_dd_pct']:>5.1f}% | {r['positive_months']}/{r['total_months']}mo+")

    # =========================================================================
    # PHASE 2: Combined optimization - use top insights
    # =========================================================================
    print("\n" + "=" * 110)
    print("PHASE 2: COMBINED OPTIMIZATION")
    print("=" * 110)

    combo_sets = []

    # Build combos from top findings
    # Base: TKY+LDN seems best session, Mon-Thu best days, exclude bad hours, exclude VAH target
    best_base = {
        'va_percent': 0.70,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3],
        'excluded_hours': [0, 10],
        'exclude_vah_target': True,
        'use_trailing': True,
        'use_prev_day': True,
        'use_prev_week': True,
    }

    # Sweep wait_candles x min_rr x sl_offset x poc_strength x tp1_rr on best base
    for wait in [3, 4, 5, 6]:
        for min_rr in [2.0, 2.5, 3.0]:
            for sl_off in [0.5, 1.0, 1.5]:
                for poc in [2.5, 3.0, 3.5]:
                    for tp1 in [1.0, 1.3]:
                        name = f"W{wait}_R{min_rr}_S{sl_off}_P{poc}_T{tp1}"
                        p = {**best_base, 'wait_candles': wait, 'min_rr': min_rr,
                             'sl_offset': sl_off, 'min_poc_strength': poc,
                             'tp1_rr': tp1, 'tp1_split': 0.5}
                        combo_sets.append((name, p))

    print(f"\n[PHASE 2] Running {len(combo_sets)} combined parameter tests...")
    print(f"{'#':>4} | {'NAME':<30} | {'TRADES':>6} | {'WR%':>6} | {'PnL R':>8} | {'PnL%':>8} | {'PF':>6} | {'MaxDD%':>7} | {'AvgR':>6} | {'TIME':>6}")
    print("-" * 125)

    combo_results = []
    for i, (name, params) in enumerate(combo_sets):
        va = params.get('va_percent', 0.70)
        data = va_data.get(va, candle_data_70)
        t0 = time.time()
        r = fast_backtest(data, params)
        elapsed = time.time() - t0

        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        if (i + 1) % 20 == 0 or r['total_pnl_r'] > 50:
            print(f"{i+1:>4} | {name:<30} | {r['total_trades']:>6} | {r['win_rate']:>5.1f}% | {r['total_pnl_r']:>+7.1f}R | {r['pnl_pct']:>+7.1f}% | {pf_str:>6} | {r['max_dd_pct']:>6.1f}% | {r['avg_pnl_r']:>+5.2f} | {elapsed:>5.1f}s")
        combo_results.append((name, params, r))

    # =========================================================================
    # PHASE 2 TOP 15
    # =========================================================================
    print(f"\n   ... {len(combo_results)} combinations tested")
    print("\n" + "=" * 110)
    print("PHASE 2 - TOP 15 BY PnL R (min 30 trades)")
    print("=" * 110)
    valid_c = [(n, p, r) for n, p, r in combo_results if r['total_trades'] >= 30]
    valid_c.sort(key=lambda x: x[2]['total_pnl_r'], reverse=True)
    for i, (name, params, r) in enumerate(valid_c[:15]):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(f"  {i+1:>2}. {name:<30} | {r['total_trades']:>5} tr | WR:{r['win_rate']:>5.1f}% | PnL:{r['total_pnl_r']:>+7.1f}R ({r['pnl_pct']:>+6.1f}%) | PF:{pf_str:>5} | DD:{r['max_dd_pct']:>5.1f}% | {r['positive_months']}/{r['total_months']}mo+")

    # TOP by PF
    print("\n" + "=" * 110)
    print("PHASE 2 - TOP 15 BY PROFIT FACTOR (min 50 trades)")
    print("=" * 110)
    valid_pf = [(n, p, r) for n, p, r in combo_results if r['total_trades'] >= 50 and r['profit_factor'] != float('inf')]
    valid_pf.sort(key=lambda x: x[2]['profit_factor'], reverse=True)
    for i, (name, params, r) in enumerate(valid_pf[:15]):
        print(f"  {i+1:>2}. {name:<30} | {r['total_trades']:>5} tr | WR:{r['win_rate']:>5.1f}% | PnL:{r['total_pnl_r']:>+7.1f}R ({r['pnl_pct']:>+6.1f}%) | PF:{r['profit_factor']:.2f} | DD:{r['max_dd_pct']:>5.1f}%")

    # =========================================================================
    # PHASE 3: Fine-tune top 3 from Phase 2
    # =========================================================================
    if valid_c:
        print("\n" + "=" * 110)
        print("PHASE 3: FINE-TUNING TOP CONFIGS")
        print("=" * 110)

        fine_results = []
        for rank_idx, (top_name, top_params, top_r) in enumerate(valid_c[:3]):
            print(f"\n--- Fine-tuning #{rank_idx+1}: {top_name} (PnL: {top_r['total_pnl_r']:+.1f}R) ---")
            # Vary tp1_split and use_trailing around this config
            for split in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for trail in [True, False]:
                    for cd in [False, True]:
                        name = f"FT{rank_idx+1}_sp{int(split*100)}_tr{int(trail)}_cd{int(cd)}"
                        p = {**top_params, 'tp1_split': split, 'use_trailing': trail}
                        if cd:
                            p['use_cooldown'] = True
                            p['cooldown_minutes'] = 60
                        va = p.get('va_percent', 0.70)
                        data = va_data.get(va, candle_data_70)
                        r = fast_backtest(data, p)
                        fine_results.append((name, p, r))

            # Also try with all sessions but keep other params
            for sess_combo in [
                {'TOKYO': True, 'LONDON': True, 'NY': False},
                {'TOKYO': True, 'LONDON': False, 'NY': False},
                {'TOKYO': True, 'LONDON': True, 'NY': True},
            ]:
                name = f"FT{rank_idx+1}_sess{''.join(k[0] for k,v in sess_combo.items() if v)}"
                p = {**top_params, 'sessions': sess_combo}
                va = p.get('va_percent', 0.70)
                data = va_data.get(va, candle_data_70)
                r = fast_backtest(data, p)
                fine_results.append((name, p, r))

            # Try other VA with same params
            for va in [0.60, 0.65, 0.75]:
                name = f"FT{rank_idx+1}_VA{int(va*100)}"
                p = {**top_params, 'va_percent': va}
                data = va_data.get(va, candle_data_70)
                r = fast_backtest(data, p)
                fine_results.append((name, p, r))

        print(f"\n   {len(fine_results)} fine-tuning tests completed")

        print("\n" + "=" * 110)
        print("PHASE 3 - TOP 15 FINE-TUNED (min 30 trades)")
        print("=" * 110)
        valid_f = [(n, p, r) for n, p, r in fine_results if r['total_trades'] >= 30]
        valid_f.sort(key=lambda x: x[2]['total_pnl_r'], reverse=True)
        for i, (name, params, r) in enumerate(valid_f[:15]):
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
            print(f"  {i+1:>2}. {name:<30} | {r['total_trades']:>5} tr | WR:{r['win_rate']:>5.1f}% | PnL:{r['total_pnl_r']:>+7.1f}R ({r['pnl_pct']:>+6.1f}%) | PF:{pf_str:>5} | DD:{r['max_dd_pct']:>5.1f}% | {r['positive_months']}/{r['total_months']}mo+")

        # =====================================================================
        # GRAND FINAL
        # =====================================================================
        all_combined = all_results + combo_results + fine_results
        print("\n" + "=" * 110)
        print("GRAND FINAL - TOP 20 OVERALL (min 30 trades)")
        print("=" * 110)
        valid_all = [(n, p, r) for n, p, r in all_combined if r['total_trades'] >= 30]
        valid_all.sort(key=lambda x: x[2]['total_pnl_r'], reverse=True)
        for i, (name, params, r) in enumerate(valid_all[:20]):
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
            print(f"  {i+1:>2}. {name:<30} | {r['total_trades']:>5} tr | WR:{r['win_rate']:>5.1f}% | PnL:{r['total_pnl_r']:>+7.1f}R ({r['pnl_pct']:>+6.1f}%) | PF:{pf_str:>5} | DD:{r['max_dd_pct']:>5.1f}% | {r['positive_months']}/{r['total_months']}mo+")

        # Best balanced (PnL > 0 and DD < 30%)
        print("\n" + "=" * 110)
        print("BEST BALANCED (PnL > 0 AND MaxDD < 30%, min 50 trades)")
        print("=" * 110)
        balanced = [(n, p, r) for n, p, r in all_combined if r['total_trades'] >= 50 and r['total_pnl_r'] > 0 and r['max_dd_pct'] < 30]
        balanced.sort(key=lambda x: x[2]['total_pnl_r'] / max(x[2]['max_dd_pct'], 1), reverse=True)
        for i, (name, params, r) in enumerate(balanced[:15]):
            ra = r['total_pnl_r'] / max(r['max_dd_pct'], 1)
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
            print(f"  {i+1:>2}. {name:<30} | {r['total_trades']:>5} tr | WR:{r['win_rate']:>5.1f}% | PnL:{r['total_pnl_r']:>+7.1f}R ({r['pnl_pct']:>+6.1f}%) | PF:{pf_str:>5} | DD:{r['max_dd_pct']:>5.1f}% | RA:{ra:.2f}")

        # Print absolute best config
        if valid_all:
            best_name, best_params, best_r = valid_all[0]
            print(f"\n{'=' * 110}")
            print(f"ABSOLUTE BEST: {best_name}")
            print(f"{'=' * 110}")
            for k, v in sorted(best_params.items()):
                print(f"  {k}: {v}")
            print(f"\n  RESULTS:")
            for k, v in sorted(best_r.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")
                else:
                    print(f"    {k}: {v}")

        if balanced:
            best_name, best_params, best_r = balanced[0]
            print(f"\n{'=' * 110}")
            print(f"BEST BALANCED: {best_name}")
            print(f"{'=' * 110}")
            for k, v in sorted(best_params.items()):
                print(f"  {k}: {v}")
            print(f"\n  RESULTS:")
            for k, v in sorted(best_r.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")
                else:
                    print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
