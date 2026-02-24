"""
Optimizer for Combined MR + CB (Mean Reversion + Confirmed Breakout) strategy.
Anti-overfitting: Train/Test split, composite scoring, stability checks.
Pattern from optimize_confirmed_breakout.py; logic from backtest_combined_mr_breakout.py.
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
from itertools import product

warnings.filterwarnings('ignore')
load_dotenv()

# =============================================================================
# FIXED CONFIG
# =============================================================================
START_DATE_STR = "2025-01-01 00:00:00"
INITIAL_CAPITAL = 1000
RISK_PERCENT = 0.001  # 0.1% risk per trade (prop firm safe)
MAX_DD_HARD_CAP = 10.0  # Prop firm constraint: reject ANY config above this

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

# Train/Test split dates
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-09-30"
TEST_START = "2025-10-01"
TEST_END = "2026-02-28"


# =============================================================================
# VP classes (reused from optimize_confirmed_breakout.py)
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
# STEP 1: Load data (reused)
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
# STEP 2: Precompute candle-level data (reused, with breakout_time tracking)
# =============================================================================
def precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70):
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
# STEP 3: Fast backtest combined MR + CB on precomputed data
# =============================================================================
def fast_backtest_combined(candle_data, params):
    """
    Combined MR+CB state machine on precomputed candle data.
    Matches backtest_combined_mr_breakout.py logic exactly.
    Single active_trade per asset - no opposing trades simultaneously.
    """
    # Shared params
    wait_candles = params.get('wait_candles', 3)
    sessions_enabled = params.get('sessions', {'TOKYO': True, 'LONDON': True, 'NY': False})
    allowed_days = set(params.get('allowed_days', [0, 1, 2, 3]))

    # MR params
    enable_mr = params.get('enable_mr', True)
    mr_min_rr = params.get('mr_min_rr', 2.0)
    mr_sl_offset = params.get('mr_sl_offset', 0.50)
    mr_tp1_rr = params.get('mr_tp1_rr', 1.3)
    mr_tp1_split = params.get('mr_tp1_split', 0.5)
    mr_tp2_split = 1.0 - mr_tp1_split
    mr_use_trailing = params.get('mr_use_trailing', True)
    mr_min_poc_strength = params.get('mr_min_poc_strength', 2.5)
    mr_filter_entry_vs_poc = params.get('mr_filter_entry_vs_poc', True)
    mr_max_breakout_duration_min = params.get('mr_max_breakout_duration_min', 4)
    mr_excluded_hours = set(params.get('mr_excluded_hours', []))

    # CB params
    enable_cb = params.get('enable_cb', True)
    cb_min_rr = params.get('cb_min_rr', 2.0)
    cb_sl_offset = params.get('cb_sl_offset', 1.0)
    cb_tp1_rr = params.get('cb_tp1_rr', 1.0)
    cb_tp1_split = params.get('cb_tp1_split', 0.3)
    cb_tp2_split = 1.0 - cb_tp1_split
    cb_use_trailing = params.get('cb_use_trailing', True)
    cb_min_poc_strength = params.get('cb_min_poc_strength', 3.0)
    cb_excluded_hours = set(params.get('cb_excluded_hours', [0, 10]))
    cb_exclude_vah_target = params.get('cb_exclude_vah_target', True)
    cb_use_prev_day = params.get('cb_use_prev_day', True)
    cb_use_prev_week = params.get('cb_use_prev_week', True)

    # State
    state = "INSIDE"
    active_trade = None
    breakout_direction = None
    breakout_time = None
    candles_since_breakout = 0
    wait_highs = []
    wait_lows = []

    current_capital = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    max_dd_percent = 0.0

    # Combined tracking
    total_trades = 0
    wins = 0
    be_trades = 0
    losses = 0
    total_pnl_r = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    # Per-strategy tracking
    mr_trades = 0
    mr_wins = 0
    mr_be = 0
    mr_losses = 0
    mr_pnl_r = 0.0
    mr_gross_profit = 0.0
    mr_gross_loss = 0.0

    cb_trades = 0
    cb_wins = 0
    cb_be = 0
    cb_losses = 0
    cb_pnl_r = 0.0
    cb_gross_profit = 0.0
    cb_gross_loss = 0.0

    # Monthly tracking
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
            breakout_time = None
            candles_since_breakout = 0
            wait_highs = []
            wait_lows = []

        # ── Manage active trade ──
        if active_trade:
            res = None
            strat = active_trade['strategy']

            if strat == 'MR':
                use_trailing = mr_use_trailing
                tp1_rr = mr_tp1_rr
                tp1_split = mr_tp1_split
                tp2_split = mr_tp2_split
            else:
                use_trailing = cb_use_trailing
                tp1_rr = cb_tp1_rr
                tp1_split = cb_tp1_split
                tp2_split = cb_tp2_split

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
                    if strat == 'MR':
                        mr_wins += 1
                    else:
                        cb_wins += 1
                else:
                    if use_trailing and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r
                        pnl = risk_amount * pnl_r
                        be_trades += 1
                        if strat == 'MR':
                            mr_be += 1
                        else:
                            cb_be += 1
                    else:
                        pnl = -risk_amount
                        pnl_r = -1.0
                        losses += 1
                        if strat == 'MR':
                            mr_losses += 1
                        else:
                            cb_losses += 1

                current_capital += pnl
                total_trades += 1
                total_pnl_r += pnl_r
                monthly_pnl[month_key] += pnl_r

                if pnl > 0:
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)

                # Per-strategy
                if strat == 'MR':
                    mr_trades += 1
                    mr_pnl_r += pnl_r
                    if pnl > 0:
                        mr_gross_profit += pnl
                    else:
                        mr_gross_loss += abs(pnl)
                else:
                    cb_trades += 1
                    cb_pnl_r += pnl_r
                    if pnl > 0:
                        cb_gross_profit += pnl
                    else:
                        cb_gross_loss += abs(pnl)

                if current_capital > high_water_mark:
                    high_water_mark = current_capital
                dd_pct = ((high_water_mark - current_capital) / high_water_mark * 100) if high_water_mark > 0 else 0
                if dd_pct > max_dd_percent:
                    max_dd_percent = dd_pct

                active_trade = None
            else:
                continue

        # ── State machine ──
        if not sessions_enabled.get(curr_sess, False):
            state = "INSIDE"
            breakout_direction = None
            candles_since_breakout = 0
            continue

        if poc is None:
            continue

        if state == "INSIDE":
            if close > vah:
                state = "BREAKOUT"
                breakout_direction = "UP"
                breakout_time = dt
                candles_since_breakout = 1
                wait_highs = [high]
                wait_lows = [low]
            elif close < val:
                state = "BREAKOUT"
                breakout_direction = "DOWN"
                breakout_time = dt
                candles_since_breakout = 1
                wait_highs = [high]
                wait_lows = [low]

        elif state == "BREAKOUT":
            candles_since_breakout += 1
            wait_highs.append(high)
            wait_lows.append(low)

            reintegrated = False
            confirmed_outside = False

            # Check reintegration (MR uses strict, CB uses non-strict)
            if enable_mr:
                if breakout_direction == "UP" and close < vah:
                    reintegrated = True
                elif breakout_direction == "DOWN" and close > val:
                    reintegrated = True
            else:
                if breakout_direction == "UP" and close <= vah:
                    reintegrated = True
                elif breakout_direction == "DOWN" and close >= val:
                    reintegrated = True

            # Check confirmed breakout
            if not reintegrated and candles_since_breakout >= wait_candles:
                confirmed_outside = True

            # ── REINTEGRATION → attempt MR ──
            if reintegrated and enable_mr:
                if breakout_direction == "UP":
                    direction = 'SHORT'
                else:
                    direction = 'LONG'

                can_time = c['can_trade']
                day_ok = c['day_of_week'] in allowed_days
                hour_ok = c['hour'] not in mr_excluded_hours

                mr_poc_ok = poc_strength is not None and poc_strength >= mr_min_poc_strength

                mr_duration_ok = True
                if breakout_time is not None:
                    bo_dur = (dt - breakout_time).total_seconds() / 60.0
                    if bo_dur >= mr_max_breakout_duration_min:
                        mr_duration_ok = False

                if mr_poc_ok and mr_duration_ok and can_time and day_ok and hour_ok:
                    if direction == 'SHORT':
                        swing_high = max(wait_highs)
                        sl = swing_high + mr_sl_offset
                        risk = sl - close
                    else:
                        swing_low = min(wait_lows)
                        sl = swing_low - mr_sl_offset
                        risk = close - sl

                    if risk > 0:
                        tp = poc
                        if direction == 'SHORT':
                            actual_rr = (close - tp) / risk
                        else:
                            actual_rr = (tp - close) / risk

                        poc_filter_ok = True
                        if mr_filter_entry_vs_poc:
                            if direction == 'SHORT' and close < poc:
                                poc_filter_ok = False
                            elif direction == 'LONG' and close > poc:
                                poc_filter_ok = False

                        rr_ok = actual_rr >= mr_min_rr and actual_rr <= 30

                        tp_in_va = True
                        if direction == 'SHORT' and tp < val:
                            tp_in_va = False
                        if direction == 'LONG' and tp > vah:
                            tp_in_va = False

                        if rr_ok and poc_filter_ok and tp_in_va:
                            active_trade = {
                                'type': direction, 'strategy': 'MR',
                                'entry': close, 'sl': sl, 'risk': risk,
                                'tp': tp, 'rr': actual_rr,
                                'partial_closed': False, 'partial_pnl_r': 0,
                            }

                # Reset state after MR attempt
                state = "INSIDE"
                breakout_direction = None
                candles_since_breakout = 0
                wait_highs = []
                wait_lows = []

            # ── CONFIRMED OUTSIDE → attempt CB ──
            elif confirmed_outside and enable_cb:
                if breakout_direction == "UP":
                    direction = 'LONG'
                else:
                    direction = 'SHORT'

                can_time = c['can_trade']
                day_ok = c['day_of_week'] in allowed_days
                hour_ok = c['hour'] not in cb_excluded_hours

                cb_poc_ok = poc_strength is not None and poc_strength >= cb_min_poc_strength

                if cb_poc_ok and can_time and day_ok and hour_ok:
                    if direction == 'LONG':
                        swing_low = min(wait_lows)
                        sl = swing_low - cb_sl_offset
                        risk = close - sl
                    else:
                        swing_high = max(wait_highs)
                        sl = swing_high + cb_sl_offset
                        risk = sl - close

                    if risk > 0:
                        # Build structural targets
                        candidates = []
                        if cb_use_prev_day:
                            if c['pd_vah'] is not None and not cb_exclude_vah_target:
                                candidates.append(('PD_VAH', c['pd_vah']))
                            if c['pd_val'] is not None:
                                candidates.append(('PD_VAL', c['pd_val']))
                            if c['pd_poc'] is not None:
                                candidates.append(('PD_POC', c['pd_poc']))
                        if cb_use_prev_week:
                            if c['pw_vah'] is not None and not cb_exclude_vah_target:
                                candidates.append(('PW_VAH', c['pw_vah']))
                            if c['pw_val'] is not None:
                                candidates.append(('PW_VAL', c['pw_val']))
                            if c['pw_poc'] is not None:
                                candidates.append(('PW_POC', c['pw_poc']))

                        if cb_exclude_vah_target:
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
                            if rr >= cb_min_rr and rr <= 30:
                                tp = price
                                break

                        if tp is None:
                            # Fallback: fixed RR
                            if direction == 'LONG':
                                tp = close + (risk * cb_min_rr)
                            else:
                                tp = close - (risk * cb_min_rr)
                            actual_rr = cb_min_rr
                        else:
                            if direction == 'LONG':
                                actual_rr = (tp - close) / risk
                            else:
                                actual_rr = (close - tp) / risk

                        active_trade = {
                            'type': direction, 'strategy': 'CB',
                            'entry': close, 'sl': sl, 'risk': risk,
                            'tp': tp, 'rr': actual_rr,
                            'partial_closed': False, 'partial_pnl_r': 0,
                        }

                # Reset state after CB attempt
                state = "INSIDE"
                breakout_direction = None
                candles_since_breakout = 0
                wait_highs = []
                wait_lows = []

            elif reintegrated and not enable_mr:
                state = "INSIDE"
                breakout_direction = None
                candles_since_breakout = 0
                wait_highs = []
                wait_lows = []

            elif confirmed_outside and not enable_cb:
                # CB disabled: stay in BREAKOUT, keep waiting for reintegration
                pass

    # Results
    win_rate = (wins + be_trades) / total_trades * 100 if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_pnl = current_capital - INITIAL_CAPITAL
    pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100

    # Monthly consistency
    positive_months = sum(1 for v in monthly_pnl.values() if v > 0)
    total_months = len(monthly_pnl)
    monthly_consistency = (positive_months / total_months * 100) if total_months > 0 else 0

    # Per-strategy
    mr_wr = (mr_wins + mr_be) / mr_trades * 100 if mr_trades > 0 else 0
    mr_pf = mr_gross_profit / mr_gross_loss if mr_gross_loss > 0 else float('inf')
    cb_wr = (cb_wins + cb_be) / cb_trades * 100 if cb_trades > 0 else 0
    cb_pf = cb_gross_profit / cb_gross_loss if cb_gross_loss > 0 else float('inf')

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
        'capital_final': current_capital,
        'monthly_consistency': monthly_consistency,
        'positive_months': positive_months,
        'total_months': total_months,
        # Per-strategy
        'mr_trades': mr_trades,
        'mr_pnl_r': mr_pnl_r,
        'mr_wr': mr_wr,
        'mr_pf': mr_pf,
        'cb_trades': cb_trades,
        'cb_pnl_r': cb_pnl_r,
        'cb_wr': cb_wr,
        'cb_pf': cb_pf,
    }


# =============================================================================
# Run on a specific date range
# =============================================================================
def run_on_period(candle_data, start_date, end_date, params):
    """Filter candle_data to date range and run fast_backtest_combined."""
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = [c for c in candle_data if start_dt <= c['dt'] <= end_dt]
    if not filtered:
        return None
    return fast_backtest_combined(filtered, params)


# =============================================================================
# Composite score for ranking
# =============================================================================
def composite_score(r):
    """
    Prop-firm optimized composite score — DD is king.
    Heavily penalizes DD, rewards PF and consistency.
    Configs above MAX_DD_HARD_CAP get score = -999.

    Score = (MAX_DD_CAP - MaxDD%) * 5  [DD margin, heavily weighted]
           + PF * 20                    [quality of trades]
           + PnL_R * 0.15              [modest PnL weight to avoid overfit]
           + monthly_consistency * 0.3  [stability]
    """
    if r is None or r['total_trades'] == 0:
        return -999
    if r['max_dd_pct'] > MAX_DD_HARD_CAP:
        return -999
    pf = r['profit_factor'] if r['profit_factor'] != float('inf') else 5.0
    pf = min(pf, 5.0)
    return (
        (MAX_DD_HARD_CAP - r['max_dd_pct']) * 5
        + pf * 20
        + r['total_pnl_r'] * 0.15
        + r['monthly_consistency'] * 0.3
    )


# =============================================================================
# Display helpers
# =============================================================================
def fmt_pf(pf):
    return f"{pf:.2f}" if pf != float('inf') else "inf"

def print_result_row(i, name, r, elapsed=None):
    t_str = f"{elapsed:>5.1f}s" if elapsed else ""
    print(f"{i:>4} | {name:<35} | {r['total_trades']:>5} | {r['win_rate']:>5.1f}% | {r['total_pnl_r']:>+7.1f}R | {r['pnl_pct']:>+7.1f}% | {fmt_pf(r['profit_factor']):>6} | {r['max_dd_pct']:>5.1f}% | {r['monthly_consistency']:>5.0f}% | MR:{r['mr_trades']:>4}/{r['mr_pnl_r']:>+6.1f}R CB:{r['cb_trades']:>4}/{r['cb_pnl_r']:>+6.1f}R {t_str}")

def print_header():
    print(f"{'#':>4} | {'NAME':<35} | {'TRDS':>5} | {'WR%':>6} | {'PnL R':>8} | {'PnL%':>8} | {'PF':>6} | {'DD%':>6} | {'CON%':>6} | {'BREAKDOWN':<30}")
    print("-" * 160)

def print_ranked(label, results, min_trades=30, top_n=15, sort_key='composite'):
    print(f"\n{'=' * 160}")
    print(label)
    print(f"{'=' * 160}")
    valid = [(n, p, r) for n, p, r in results if r['total_trades'] >= min_trades]
    if sort_key == 'composite':
        valid.sort(key=lambda x: composite_score(x[2]), reverse=True)
    elif sort_key == 'pnl':
        valid.sort(key=lambda x: x[2]['total_pnl_r'], reverse=True)
    elif sort_key == 'pf':
        valid.sort(key=lambda x: x[2]['profit_factor'] if x[2]['profit_factor'] != float('inf') else 999, reverse=True)
    print_header()
    for i, (name, params, r) in enumerate(valid[:top_n]):
        cs = composite_score(r)
        print_result_row(i+1, f"{name} [CS:{cs:.1f}]", r)
    return valid


# =============================================================================
# BASELINE PARAMS (matching backtest_combined_mr_breakout.py defaults)
# =============================================================================
BASELINE = {
    'wait_candles': 3,
    'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
    'allowed_days': [0, 1, 2, 3],
    # MR
    'enable_mr': True,
    'mr_min_rr': 2.0,
    'mr_sl_offset': 0.50,
    'mr_tp1_rr': 1.3,
    'mr_tp1_split': 0.5,
    'mr_use_trailing': True,
    'mr_min_poc_strength': 2.5,
    'mr_filter_entry_vs_poc': True,
    'mr_max_breakout_duration_min': 4,
    'mr_excluded_hours': [],
    # CB
    'enable_cb': True,
    'cb_min_rr': 2.0,
    'cb_sl_offset': 1.0,
    'cb_tp1_rr': 1.0,
    'cb_tp1_split': 0.3,
    'cb_use_trailing': True,
    'cb_min_poc_strength': 3.0,
    'cb_excluded_hours': [0, 10],
    'cb_exclude_vah_target': True,
    'cb_use_prev_day': True,
    'cb_use_prev_week': True,
}


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 160)
    print("OPTIMIZER - Combined MR + CB (Anti-Overfitting: Train/Test Split)")
    print("=" * 160)
    print(f"  Train: {TRAIN_START} to {TRAIN_END} | Test: {TEST_START} to {TEST_END}")
    print(f"  Risk: {RISK_PERCENT*100}% per trade | Max DD cap: {MAX_DD_HARD_CAP}%")
    print(f"  Composite = (DDcap-DD)*5 + PF*20 + PnL_R*0.15 + Consistency*0.3")

    # =====================================================================
    # Load data
    # =====================================================================
    df_candles, ticks_by_minute, requested_start = load_data()

    # Precompute for VA = 0.65, 0.70, 0.75
    va_data = {}
    for va in [0.65, 0.70, 0.75]:
        print(f"\n[PRECOMPUTE] Building VP levels for VA={int(va*100)}%...")
        t0 = time.time()
        va_data[va] = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=va)
        print(f"   {len(va_data[va]):,} candles in {time.time()-t0:.1f}s")

    candle_data = va_data[0.70]  # default

    # =====================================================================
    # Verify baseline on full period
    # =====================================================================
    print("\n[VERIFY] Running baseline on full period...")
    t0 = time.time()
    baseline_r = fast_backtest_combined(candle_data, BASELINE)
    print(f"   Baseline: {baseline_r['total_trades']} trades | PnL: {baseline_r['total_pnl_r']:+.1f}R ({baseline_r['pnl_pct']:+.1f}%) | PF: {fmt_pf(baseline_r['profit_factor'])} | DD: {baseline_r['max_dd_pct']:.1f}% | {baseline_r['positive_months']}/{baseline_r['total_months']} mo+ | {time.time()-t0:.1f}s")
    print(f"   MR: {baseline_r['mr_trades']} trades, {baseline_r['mr_pnl_r']:+.1f}R, WR:{baseline_r['mr_wr']:.1f}%, PF:{fmt_pf(baseline_r['mr_pf'])}")
    print(f"   CB: {baseline_r['cb_trades']} trades, {baseline_r['cb_pnl_r']:+.1f}R, WR:{baseline_r['cb_wr']:.1f}%, PF:{fmt_pf(baseline_r['cb_pf'])}")

    # =====================================================================
    # PHASE 1: Individual Parameter Sweeps (full period)
    # =====================================================================
    print("\n" + "=" * 160)
    print("PHASE 1: INDIVIDUAL PARAMETER SWEEPS (full period)")
    print("=" * 160)

    phase1 = []
    phase1.append(("BASELINE", {**BASELINE}))

    # Shared: wait_candles
    for w in [2, 4, 5, 6]:
        phase1.append((f"wait_{w}", {**BASELINE, 'wait_candles': w}))

    # Shared: sessions
    phase1.append(("sess_TKY+LDN+NY", {**BASELINE, 'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True}}))
    phase1.append(("sess_TKY_only", {**BASELINE, 'sessions': {'TOKYO': True, 'LONDON': False, 'NY': False}}))
    phase1.append(("sess_LDN_only", {**BASELINE, 'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False}}))

    # Shared: allowed_days
    phase1.append(("days_Mon-Fri", {**BASELINE, 'allowed_days': [0, 1, 2, 3, 4]}))
    phase1.append(("days_Tue-Thu", {**BASELINE, 'allowed_days': [1, 2, 3]}))
    phase1.append(("days_MonTueWedFri", {**BASELINE, 'allowed_days': [0, 1, 2, 4]}))

    # MR: min_rr
    for v in [1.5, 2.5, 3.0]:
        phase1.append((f"mr_rr_{v}", {**BASELINE, 'mr_min_rr': v}))

    # MR: sl_offset
    for v in [0.25, 1.0, 1.5]:
        phase1.append((f"mr_sl_{v}", {**BASELINE, 'mr_sl_offset': v}))

    # MR: tp1_rr
    for v in [0.8, 1.0, 1.5]:
        phase1.append((f"mr_tp1rr_{v}", {**BASELINE, 'mr_tp1_rr': v}))

    # MR: tp1_split
    for v in [0.3, 0.7]:
        phase1.append((f"mr_split_{int(v*100)}", {**BASELINE, 'mr_tp1_split': v}))

    # MR: poc_strength
    for v in [2.0, 3.0, 3.5]:
        phase1.append((f"mr_poc_{v}", {**BASELINE, 'mr_min_poc_strength': v}))

    # MR: max_breakout_duration
    for v in [2, 3, 6, 10]:
        phase1.append((f"mr_dur_{v}m", {**BASELINE, 'mr_max_breakout_duration_min': v}))

    # MR: filter_entry_vs_poc
    phase1.append(("mr_noPocFilter", {**BASELINE, 'mr_filter_entry_vs_poc': False}))

    # MR: excluded_hours
    phase1.append(("mr_excH_0", {**BASELINE, 'mr_excluded_hours': [0]}))
    phase1.append(("mr_excH_0_3", {**BASELINE, 'mr_excluded_hours': [0, 3]}))

    # CB: min_rr
    for v in [1.5, 2.5, 3.0]:
        phase1.append((f"cb_rr_{v}", {**BASELINE, 'cb_min_rr': v}))

    # CB: sl_offset
    for v in [0.5, 1.5, 2.0]:
        phase1.append((f"cb_sl_{v}", {**BASELINE, 'cb_sl_offset': v}))

    # CB: tp1_rr
    for v in [0.5, 1.3, 1.5]:
        phase1.append((f"cb_tp1rr_{v}", {**BASELINE, 'cb_tp1_rr': v}))

    # CB: tp1_split
    for v in [0.5, 0.7]:
        phase1.append((f"cb_split_{int(v*100)}", {**BASELINE, 'cb_tp1_split': v}))

    # CB: poc_strength
    for v in [2.0, 2.5, 3.5]:
        phase1.append((f"cb_poc_{v}", {**BASELINE, 'cb_min_poc_strength': v}))

    # CB: excluded_hours
    phase1.append(("cb_noExcH", {**BASELINE, 'cb_excluded_hours': []}))
    phase1.append(("cb_excH_0_10_15", {**BASELINE, 'cb_excluded_hours': [0, 10, 15]}))

    # CB: exclude_vah_target
    phase1.append(("cb_allowVAH", {**BASELINE, 'cb_exclude_vah_target': False}))

    print(f"\nRunning {len(phase1)} configs...")
    print_header()

    phase1_results = []
    for i, (name, params) in enumerate(phase1):
        t0 = time.time()
        r = fast_backtest_combined(candle_data, params)
        elapsed = time.time() - t0
        print_result_row(i+1, name, r, elapsed)
        phase1_results.append((name, params, r))

    # Phase 1 rankings
    print_ranked("PHASE 1 - TOP 10 BY PnL R", phase1_results, min_trades=30, top_n=10, sort_key='pnl')
    print_ranked("PHASE 1 - TOP 10 BY COMPOSITE SCORE", phase1_results, min_trades=30, top_n=10, sort_key='composite')

    # =====================================================================
    # PHASE 2: Combined Grid (TRAIN period only)
    # =====================================================================
    print("\n" + "=" * 160)
    print("PHASE 2: COMBINED GRID (TRAIN PERIOD ONLY)")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print("=" * 160)

    # Build grid from coarse sweeps
    phase2 = []

    # MR grid
    mr_rr_vals = [1.5, 2.0, 2.5]
    mr_sl_vals = [0.25, 0.50, 1.0]
    mr_poc_vals = [2.0, 2.5, 3.0]
    mr_dur_vals = [3, 4, 6]

    # CB grid
    cb_rr_vals = [1.5, 2.0, 2.5]
    cb_sl_vals = [0.5, 1.0, 1.5]
    cb_poc_vals = [2.5, 3.0, 3.5]

    # Shared grid
    wait_vals = [3, 4, 5]
    sess_vals = [
        {'TOKYO': True, 'LONDON': True, 'NY': False},
        {'TOKYO': True, 'LONDON': True, 'NY': True},
    ]
    days_vals = [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    ]

    # Build combos: sweep MR x CB x Shared
    # To keep it manageable (~300), we combine:
    # - 3 MR_RR x 3 MR_SL x 3 MR_POC = 27 MR combos (fix dur=4)
    # - 3 CB_RR x 3 CB_SL x 3 CB_POC = 27 CB combos
    # - 3 wait x 2 sess x 2 days = 12 shared combos
    # Total would be 27*27*12 = too many. Instead:
    # Pick top-3 MR combos x top-3 CB combos x all shared = 3*3*12 = 108
    # Plus: all MR combos with best shared, all CB combos with best shared

    # First: sweep MR combos on train (fix CB and shared to baseline)
    print("\n[PHASE 2a] Sweeping MR parameter combos on TRAIN...")
    mr_combos = []
    for mr_rr, mr_sl, mr_poc, mr_dur in product(mr_rr_vals, mr_sl_vals, mr_poc_vals, mr_dur_vals):
        p = {**BASELINE, 'mr_min_rr': mr_rr, 'mr_sl_offset': mr_sl,
             'mr_min_poc_strength': mr_poc, 'mr_max_breakout_duration_min': mr_dur}
        name = f"MR_rr{mr_rr}_sl{mr_sl}_poc{mr_poc}_dur{mr_dur}"
        t0 = time.time()
        r = run_on_period(candle_data, TRAIN_START, TRAIN_END, p)
        if r and r['total_trades'] >= 20:
            mr_combos.append((name, p, r))
    mr_combos.sort(key=lambda x: composite_score(x[2]), reverse=True)
    print(f"   {len(mr_combos)} MR combos tested")
    for i, (n, p, r) in enumerate(mr_combos[:5]):
        print(f"   {i+1}. {n} | {r['total_trades']} tr | PnL:{r['total_pnl_r']:+.1f}R | PF:{fmt_pf(r['profit_factor'])} | MR:{r['mr_pnl_r']:+.1f}R | CS:{composite_score(r):.1f}")

    # Sweep CB combos on train
    print("\n[PHASE 2b] Sweeping CB parameter combos on TRAIN...")
    cb_combos = []
    for cb_rr, cb_sl, cb_poc in product(cb_rr_vals, cb_sl_vals, cb_poc_vals):
        p = {**BASELINE, 'cb_min_rr': cb_rr, 'cb_sl_offset': cb_sl,
             'cb_min_poc_strength': cb_poc}
        name = f"CB_rr{cb_rr}_sl{cb_sl}_poc{cb_poc}"
        r = run_on_period(candle_data, TRAIN_START, TRAIN_END, p)
        if r and r['total_trades'] >= 20:
            cb_combos.append((name, p, r))
    cb_combos.sort(key=lambda x: composite_score(x[2]), reverse=True)
    print(f"   {len(cb_combos)} CB combos tested")
    for i, (n, p, r) in enumerate(cb_combos[:5]):
        print(f"   {i+1}. {n} | {r['total_trades']} tr | PnL:{r['total_pnl_r']:+.1f}R | PF:{fmt_pf(r['profit_factor'])} | CB:{r['cb_pnl_r']:+.1f}R | CS:{composite_score(r):.1f}")

    # Sweep shared combos
    print("\n[PHASE 2c] Sweeping shared parameter combos on TRAIN...")
    shared_combos = []
    for wait, sess, days in product(wait_vals, sess_vals, days_vals):
        p = {**BASELINE, 'wait_candles': wait, 'sessions': sess, 'allowed_days': days}
        sess_str = '+'.join(k[0] for k, v in sess.items() if v)
        days_str = ''.join(str(d) for d in days)
        name = f"SH_w{wait}_{sess_str}_d{days_str}"
        r = run_on_period(candle_data, TRAIN_START, TRAIN_END, p)
        if r and r['total_trades'] >= 20:
            shared_combos.append((name, p, r))
    shared_combos.sort(key=lambda x: composite_score(x[2]), reverse=True)
    print(f"   {len(shared_combos)} shared combos tested")
    for i, (n, p, r) in enumerate(shared_combos[:5]):
        print(f"   {i+1}. {n} | {r['total_trades']} tr | PnL:{r['total_pnl_r']:+.1f}R | PF:{fmt_pf(r['profit_factor'])} | CS:{composite_score(r):.1f}")

    # Now cross-product: top-5 MR x top-5 CB x top-4 shared
    print("\n[PHASE 2d] Cross-product of top MR x CB x shared...")
    phase2_results = []
    top_mr = mr_combos[:5]
    top_cb = cb_combos[:5]
    top_sh = shared_combos[:4]

    count = 0
    for (mr_name, mr_p, _), (cb_name, cb_p, _), (sh_name, sh_p, _) in product(top_mr, top_cb, top_sh):
        combined_p = {**BASELINE}
        # Apply MR params
        for k in ['mr_min_rr', 'mr_sl_offset', 'mr_min_poc_strength', 'mr_max_breakout_duration_min']:
            combined_p[k] = mr_p[k]
        # Apply CB params
        for k in ['cb_min_rr', 'cb_sl_offset', 'cb_min_poc_strength']:
            combined_p[k] = cb_p[k]
        # Apply shared params
        for k in ['wait_candles', 'sessions', 'allowed_days']:
            combined_p[k] = sh_p[k]

        name = f"{mr_name}|{cb_name}|{sh_name}"
        r = run_on_period(candle_data, TRAIN_START, TRAIN_END, combined_p)
        if r and r['total_trades'] >= 30:
            phase2_results.append((name, combined_p, r))
        count += 1
        if count % 20 == 0:
            print(f"   ... {count} combos tested")

    print(f"\n   {len(phase2_results)} valid combos from {count} total")

    # Rank by composite score
    phase2_results.sort(key=lambda x: composite_score(x[2]), reverse=True)

    print_ranked("PHASE 2 - TOP 20 BY COMPOSITE SCORE (TRAIN)", phase2_results, min_trades=30, top_n=20, sort_key='composite')

    # =====================================================================
    # PHASE 3: Validate Top 20 on TEST period
    # =====================================================================
    print("\n" + "=" * 160)
    print("PHASE 3: VALIDATION ON TEST PERIOD")
    print(f"Test: {TEST_START} to {TEST_END}")
    print("=" * 160)

    # Take top 20 from train
    top20_train = phase2_results[:20]

    validated = []
    print(f"\n{'#':>3} | {'NAME':<50} | {'TRAIN':>45} | {'TEST':>45} | {'STATUS'}")
    print("-" * 200)

    for i, (name, params, train_r) in enumerate(top20_train):
        test_r = run_on_period(candle_data, TEST_START, TEST_END, params)
        if test_r is None:
            status = "NO DATA"
            continue

        # Stability check: profitable on BOTH periods
        train_ok = train_r['total_pnl_r'] > 0
        test_ok = test_r['total_pnl_r'] > 0
        test_min_trades = test_r['total_trades'] >= 15

        if train_ok and test_ok and test_min_trades:
            status = "PASS"
            test_cs = composite_score(test_r)
            validated.append((name, params, train_r, test_r, test_cs))
        elif not test_ok:
            status = "FAIL (test negative)"
        elif not test_min_trades:
            status = "FAIL (test < 15 trades)"
        else:
            status = "FAIL"

        train_str = f"{train_r['total_trades']:>4}tr {train_r['total_pnl_r']:>+7.1f}R PF:{fmt_pf(train_r['profit_factor']):>5} DD:{train_r['max_dd_pct']:>5.1f}%"
        test_str = f"{test_r['total_trades']:>4}tr {test_r['total_pnl_r']:>+7.1f}R PF:{fmt_pf(test_r['profit_factor']):>5} DD:{test_r['max_dd_pct']:>5.1f}%"
        print(f"{i+1:>3} | {name[:50]:<50} | {train_str} | {test_str} | {status}")

    # Also validate baseline
    baseline_train = run_on_period(candle_data, TRAIN_START, TRAIN_END, BASELINE)
    baseline_test = run_on_period(candle_data, TEST_START, TEST_END, BASELINE)
    if baseline_train and baseline_test:
        print(f"\n  BASELINE REFERENCE:")
        print(f"    Train: {baseline_train['total_trades']}tr {baseline_train['total_pnl_r']:+.1f}R PF:{fmt_pf(baseline_train['profit_factor'])} DD:{baseline_train['max_dd_pct']:.1f}%")
        print(f"    Test:  {baseline_test['total_trades']}tr {baseline_test['total_pnl_r']:+.1f}R PF:{fmt_pf(baseline_test['profit_factor'])} DD:{baseline_test['max_dd_pct']:.1f}%")

    # Rank validated by test composite score
    validated.sort(key=lambda x: x[4], reverse=True)

    print(f"\n{'=' * 160}")
    print(f"PHASE 3 - VALIDATED CONFIGS (profitable on BOTH train AND test)")
    print(f"{'=' * 160}")
    print(f"{'#':>3} | {'NAME':<50} | {'TRAIN PnL':>10} | {'TEST PnL':>10} | {'TRAIN PF':>9} | {'TEST PF':>8} | {'TRAIN DD':>9} | {'TEST DD':>8} | {'TEST CS':>8}")
    print("-" * 170)
    for i, (name, params, train_r, test_r, test_cs) in enumerate(validated):
        print(f"{i+1:>3} | {name[:50]:<50} | {train_r['total_pnl_r']:>+9.1f}R | {test_r['total_pnl_r']:>+9.1f}R | {fmt_pf(train_r['profit_factor']):>9} | {fmt_pf(test_r['profit_factor']):>8} | {train_r['max_dd_pct']:>8.1f}% | {test_r['max_dd_pct']:>7.1f}% | {test_cs:>7.1f}")

    # =====================================================================
    # PHASE 4: VA% Sweep on top-5 validated
    # =====================================================================
    print("\n" + "=" * 160)
    print("PHASE 4: VA% SWEEP ON TOP VALIDATED CONFIGS")
    print("=" * 160)

    phase4_results = []
    top5_validated = validated[:5]

    for vi, (name, params, train_r, test_r, _) in enumerate(top5_validated):
        print(f"\n--- Config #{vi+1}: {name[:60]} ---")
        for va in [0.65, 0.70, 0.75]:
            data = va_data[va]
            va_train = run_on_period(data, TRAIN_START, TRAIN_END, params)
            va_test = run_on_period(data, TEST_START, TEST_END, params)
            va_full = fast_backtest_combined(data, params)
            if va_train and va_test and va_full:
                train_ok = va_train['total_pnl_r'] > 0
                test_ok = va_test['total_pnl_r'] > 0
                status = "OK" if train_ok and test_ok else "FAIL"
                cs_test = composite_score(va_test)
                cs_full = composite_score(va_full)
                print(f"  VA={int(va*100)}% | Full: {va_full['total_trades']}tr {va_full['total_pnl_r']:+.1f}R PF:{fmt_pf(va_full['profit_factor'])} DD:{va_full['max_dd_pct']:.1f}% | Train: {va_train['total_pnl_r']:+.1f}R | Test: {va_test['total_pnl_r']:+.1f}R | {status}")
                if train_ok and test_ok:
                    phase4_results.append((f"VA{int(va*100)}|{name[:40]}", params, va_full, va_train, va_test, cs_full, va))

    # Final ranking
    phase4_results.sort(key=lambda x: x[5], reverse=True)

    print(f"\n{'=' * 160}")
    print("PHASE 4 - FINAL TOP 10")
    print(f"{'=' * 160}")
    print(f"{'#':>3} | {'NAME':<55} | {'VA':>3} | {'FULL PnL':>9} | {'FULL PF':>8} | {'FULL DD':>8} | {'TRAIN PnL':>10} | {'TEST PnL':>9} | {'CS':>6}")
    print("-" * 170)
    for i, (name, params, full_r, train_r, test_r, cs, va) in enumerate(phase4_results[:10]):
        print(f"{i+1:>3} | {name[:55]:<55} | {int(va*100):>3} | {full_r['total_pnl_r']:>+8.1f}R | {fmt_pf(full_r['profit_factor']):>8} | {full_r['max_dd_pct']:>7.1f}% | {train_r['total_pnl_r']:>+9.1f}R | {test_r['total_pnl_r']:>+8.1f}R | {cs:>5.1f}")

    # =====================================================================
    # FINAL RECOMMENDATION
    # =====================================================================
    print("\n" + "=" * 160)
    print("FINAL RECOMMENDATION")
    print("=" * 160)

    if phase4_results:
        best_name, best_params, best_full, best_train, best_test, best_cs, best_va = phase4_results[0]
        print(f"\n  BEST CONFIG: {best_name}")
        print(f"  VA%: {int(best_va*100)}%")
        print(f"  Composite Score: {best_cs:.1f}")
        print(f"\n  FULL PERIOD: {best_full['total_trades']} trades | PnL: {best_full['total_pnl_r']:+.1f}R ({best_full['pnl_pct']:+.1f}%) | PF: {fmt_pf(best_full['profit_factor'])} | DD: {best_full['max_dd_pct']:.1f}% | {best_full['positive_months']}/{best_full['total_months']} months+")
        print(f"    MR: {best_full['mr_trades']} trades, {best_full['mr_pnl_r']:+.1f}R, WR:{best_full['mr_wr']:.1f}%, PF:{fmt_pf(best_full['mr_pf'])}")
        print(f"    CB: {best_full['cb_trades']} trades, {best_full['cb_pnl_r']:+.1f}R, WR:{best_full['cb_wr']:.1f}%, PF:{fmt_pf(best_full['cb_pf'])}")
        print(f"\n  TRAIN: {best_train['total_trades']} trades | PnL: {best_train['total_pnl_r']:+.1f}R | PF: {fmt_pf(best_train['profit_factor'])} | DD: {best_train['max_dd_pct']:.1f}%")
        print(f"  TEST:  {best_test['total_trades']} trades | PnL: {best_test['total_pnl_r']:+.1f}R | PF: {fmt_pf(best_test['profit_factor'])} | DD: {best_test['max_dd_pct']:.1f}%")

        print(f"\n  PARAMS (copy-paste into backtest_combined_mr_breakout.py):")
        print(f"  " + "-" * 60)
        # Map back to global config names
        param_map = {
            'wait_candles': 'WAIT_CANDLES',
            'mr_min_rr': 'MR_MIN_RR',
            'mr_sl_offset': 'MR_SL_OFFSET',
            'mr_tp1_rr': 'MR_TP1_RR',
            'mr_tp1_split': 'MR_TP1_SPLIT',
            'mr_use_trailing': 'MR_USE_TRAILING',
            'mr_min_poc_strength': 'MR_MIN_POC_STRENGTH',
            'mr_filter_entry_vs_poc': 'MR_FILTER_ENTRY_VS_POC',
            'mr_max_breakout_duration_min': 'MR_MAX_BREAKOUT_DURATION_MINUTES',
            'mr_excluded_hours': 'MR_EXCLUDED_HOURS',
            'cb_min_rr': 'CB_MIN_RR',
            'cb_sl_offset': 'CB_SL_OFFSET',
            'cb_tp1_rr': 'CB_TP1_RR',
            'cb_tp1_split': 'CB_TP1_SPLIT',
            'cb_use_trailing': 'CB_USE_TRAILING',
            'cb_min_poc_strength': 'CB_MIN_POC_STRENGTH',
            'cb_excluded_hours': 'CB_EXCLUDED_HOURS',
            'cb_exclude_vah_target': 'CB_EXCLUDE_VAH_TARGET',
            'cb_use_prev_day': 'CB_USE_PREV_DAY',
            'cb_use_prev_week': 'CB_USE_PREV_WEEK',
        }
        for opt_key, cfg_key in param_map.items():
            val = best_params.get(opt_key)
            if val is not None:
                if isinstance(val, bool):
                    print(f"  {cfg_key} = {val}")
                elif isinstance(val, list):
                    print(f"  {cfg_key} = {val}")
                elif isinstance(val, float):
                    print(f"  {cfg_key} = {val}")
                elif isinstance(val, int):
                    print(f"  {cfg_key} = {val}")
                else:
                    print(f"  {cfg_key} = {repr(val)}")

        # Also print asset config
        sess = best_params.get('sessions', {})
        days = best_params.get('allowed_days', [])
        print(f"\n  # Asset config:")
        print(f"  'sessions': {sess},")
        print(f"  'allowed_days': {days},")
        print(f"  'va_percent': {best_va},")

    elif validated:
        # Fallback: use best validated without VA sweep
        best_name, best_params, best_train, best_test, best_cs = validated[0]
        best_full = fast_backtest_combined(candle_data, best_params)
        print(f"\n  BEST VALIDATED (no VA sweep improvement): {best_name}")
        print(f"  Full: {best_full['total_trades']} trades | PnL: {best_full['total_pnl_r']:+.1f}R | PF: {fmt_pf(best_full['profit_factor'])} | DD: {best_full['max_dd_pct']:.1f}%")
        print(f"  Train: {best_train['total_pnl_r']:+.1f}R | Test: {best_test['total_pnl_r']:+.1f}R")
        print(f"\n  PARAMS:")
        for k, v in sorted(best_params.items()):
            print(f"    {k}: {v}")
    else:
        print("\n  NO CONFIGS PASSED VALIDATION!")
        print("  Consider relaxing constraints or reviewing strategy logic.")
        if phase2_results:
            print(f"\n  Best TRAIN-only config for reference:")
            n, p, r = phase2_results[0]
            print(f"    {n}: {r['total_trades']} trades | PnL: {r['total_pnl_r']:+.1f}R | PF: {fmt_pf(r['profit_factor'])}")

    # Comparison vs baseline
    print(f"\n{'=' * 160}")
    print("BASELINE COMPARISON")
    print(f"{'=' * 160}")
    full_base = fast_backtest_combined(candle_data, BASELINE)
    print(f"  Baseline (full): {full_base['total_trades']} trades | PnL: {full_base['total_pnl_r']:+.1f}R ({full_base['pnl_pct']:+.1f}%) | PF: {fmt_pf(full_base['profit_factor'])} | DD: {full_base['max_dd_pct']:.1f}%")
    if phase4_results:
        _, _, best_full, _, _, _, _ = phase4_results[0]
        delta_pnl = best_full['total_pnl_r'] - full_base['total_pnl_r']
        delta_dd = best_full['max_dd_pct'] - full_base['max_dd_pct']
        print(f"  Best opt (full):  {best_full['total_trades']} trades | PnL: {best_full['total_pnl_r']:+.1f}R ({best_full['pnl_pct']:+.1f}%) | PF: {fmt_pf(best_full['profit_factor'])} | DD: {best_full['max_dd_pct']:.1f}%")
        print(f"  Delta: PnL {delta_pnl:+.1f}R | DD {delta_dd:+.1f}%")


if __name__ == "__main__":
    main()
