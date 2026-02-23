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

warnings.filterwarnings('ignore')
load_dotenv()

# =============================================================================
# STRATEGY ENABLE/DISABLE
# =============================================================================
ENABLE_MR = True          # Mean Reversion after reintegration
ENABLE_CB = True          # Confirmed Breakout after sustained breakout

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================
START_DATE_STR = "2025-01-01 00:00:00"
INITIAL_CAPITAL = 1000
RISK_PERCENT = 0.01

WAIT_CANDLES = 3          # Candles to wait after breakout before deciding MR or CB

DISPLAY_MODE = "MONTHLY"
RESET_VP_PER_SESSION = True

USE_COOLDOWN_AFTER_LOSS = False
COOLDOWN_AFTER_LOSS_MINUTES = 60

SHOW_LAST_TRADES = False
LAST_TRADES_COUNT = 10
SHOW_OPEN_TRADES = False

# =============================================================================
# MR-SPECIFIC CONFIG (Mean Reversion)
# =============================================================================
MR_TP_MODE = "POC"            # TP target for MR (POC of session VP)
MR_MIN_RR = 2.0
MR_SL_OFFSET = 0.50
MR_TP1_RR = 1.3
MR_TP1_SPLIT = 0.5
MR_TP2_SPLIT = 0.5
MR_USE_TRAILING = True
MR_USE_VP_STRUCTURE_FILTER = True
MR_MIN_POC_STRENGTH = 2.5
MR_FILTER_ENTRY_VS_POC = True
MR_USE_BREAKOUT_DURATION_FILTER = True
MR_MAX_BREAKOUT_DURATION_MINUTES = 4  # Max minutes in breakout before reintegration
MR_EXCLUDED_HOURS = []

# =============================================================================
# CB-SPECIFIC CONFIG (Confirmed Breakout)
# =============================================================================
CB_MIN_RR = 2.0
CB_SL_OFFSET = 1.0
CB_TP1_RR = 1.0
CB_TP1_SPLIT = 0.3
CB_TP2_SPLIT = 0.7
CB_USE_TRAILING = True
CB_USE_VP_STRUCTURE_FILTER = True
CB_MIN_POC_STRENGTH = 3.0
CB_EXCLUDE_VAH_TARGET = True
CB_USE_PREV_DAY = True
CB_USE_PREV_WEEK = True
CB_EXCLUDED_HOURS = [0, 10]

# =============================================================================
# SHARED FILTERS
# =============================================================================
USE_VP_SHAPE_FILTER = True
EXCLUDED_VP_SHAPES = [""]

# =============================================================================
# SESSIONS (UTC)
# =============================================================================
SESSIONS_CONFIG = {
    'TOKYO':  {'vp_start': 0,    'vp_end': 4,    'trade_start': 0,    'trade_end': 4},
    'LONDON': {'vp_start': 8,    'vp_end': 14.5, 'trade_start': 9,    'trade_end': 14},
    'NY':     {'vp_start': 14.5, 'vp_end': 21.5, 'trade_start': 15,   'trade_end': 21},
}

ASSETS = [
    {
        'enabled': True,
        'symbol': 'XAUUSD',
        'candle_table': 'candles_mt5_xauusd_1m',
        'tick_table': 'market_ticks_xauusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3],
    },
]


# =============================================================================
# DATABASE
# =============================================================================
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('PG_HOST'), port=os.getenv('PG_PORT'),
            database=os.getenv('PG_DB'), user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Erreur DB: {e}")
        sys.exit(1)


# =============================================================================
# INCREMENTAL VOLUME PROFILE
# =============================================================================
class IncrementalVolumeProfile:
    def __init__(self, tick_size: float = 0.01, va_percent: float = 0.70):
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

    def add_ticks(self, prices: np.ndarray, volumes: np.ndarray):
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

    def get_profile_shape(self):
        if not self._cache_valid:
            self._compute_value_area()
        if self._cached_poc_strength is None or self._cached_poc_strength < 1.3:
            return "FLAT"
        if self._cached_poc is None or self._cached_vah is None or self._cached_val is None:
            return "FLAT"
        va_range = self._cached_vah - self._cached_val
        if va_range == 0:
            return "FLAT"
        poc_position = (self._cached_poc - self._cached_val) / va_range
        if poc_position > 0.66:
            return "P-SHAPE"
        elif poc_position < 0.33:
            return "B-SHAPE"
        else:
            return "D-SHAPE"

    def snapshot(self):
        poc, vah, val = self.get_levels()
        return {'poc': poc, 'vah': vah, 'val': val}


# =============================================================================
# STRUCTURAL LEVELS TRACKER (for CB targets)
# =============================================================================
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

    def get_target_levels(self, direction: str, entry_price: float):
        candidates = []
        if CB_USE_PREV_DAY:
            if self.prev_day['vah'] is not None and not CB_EXCLUDE_VAH_TARGET:
                candidates.append(('PD_VAH', self.prev_day['vah']))
            if self.prev_day['val'] is not None:
                candidates.append(('PD_VAL', self.prev_day['val']))
            if self.prev_day['poc'] is not None:
                candidates.append(('PD_POC', self.prev_day['poc']))
        if CB_USE_PREV_WEEK:
            if self.prev_week['vah'] is not None and not CB_EXCLUDE_VAH_TARGET:
                candidates.append(('PW_VAH', self.prev_week['vah']))
            if self.prev_week['val'] is not None:
                candidates.append(('PW_VAL', self.prev_week['val']))
            if self.prev_week['poc'] is not None:
                candidates.append(('PW_POC', self.prev_week['poc']))
        if direction == 'LONG':
            valid = [(label, price) for label, price in candidates if price > entry_price]
            valid.sort(key=lambda x: x[1])
        else:
            valid = [(label, price) for label, price in candidates if price < entry_price]
            valid.sort(key=lambda x: -x[1])
        return valid


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_session_start_time(session_name, reference_dt):
    if session_name not in SESSIONS_CONFIG:
        return None
    cfg = SESSIONS_CONFIG[session_name]
    start_hour = cfg['vp_start']
    hour = int(start_hour)
    minute = int((start_hour % 1) * 60)
    return reference_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


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


def load_all_data(conn, asset):
    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = requested_start - timedelta(days=14)
    ts_start = int(data_start.timestamp() * 1000)

    query_candles = f"SELECT ts, open, high, low, close FROM {asset['candle_table']} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df_candles = pd.read_sql(query_candles, conn)
    df_candles['dt'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)

    if df_candles.empty:
        return df_candles, pd.DataFrame()

    t_start = data_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = df_candles['dt'].max().strftime("%Y-%m-%d %H:%M:%S")

    query_ticks = f"""SELECT time, last as price, volume FROM {asset['tick_table']}
        WHERE time >= '{t_start}' AND time <= '{t_end}'
        ORDER BY time ASC"""

    df_ticks = pd.read_sql(query_ticks, conn)
    df_ticks['time'] = pd.to_datetime(df_ticks['time'], utc=True)

    return df_candles, df_ticks


def get_exit_scenario(trade):
    result = trade['result']
    tp1_hit = pd.notna(trade.get('tp1_time'))
    tp2_hit = pd.notna(trade.get('tp2_time'))
    if result == "WIN":
        if tp1_hit and tp2_hit:
            return ("TP1_TP2", "TP1 -> TP2 (Full Win)", "++")
        elif tp2_hit:
            return ("TP2_DIRECT", "TP2 direct", "+")
        else:
            return ("WIN_OTHER", "Win (autre)", "+")
    elif result == "BE":
        if tp1_hit:
            return ("TP1_BE", "TP1 -> BE (SL@Entry)", "~")
        else:
            return ("BE_OTHER", "BE (autre)", "~")
    elif result == "LOSS":
        if tp1_hit:
            return ("TP1_SL", "TP1 -> SL (anormal)", "!")
        else:
            return ("SL_DIRECT", "SL direct (Full Loss)", "X")
    return ("UNKNOWN", "Unknown", "?")


def display_last_trades(df_trades, count):
    if df_trades.empty:
        return
    last_trades = df_trades.tail(count)
    print(f"\n{'=' * 120}")
    print(f"DETAIL DES {len(last_trades)} DERNIERS TRADES")
    print(f"{'=' * 120}")

    for idx, trade in last_trades.iterrows():
        scenario_code, scenario_label, emoji = get_exit_scenario(trade)
        strat = trade.get('strategy', '?')

        # Per-strategy params for display
        if strat == 'MR':
            tp1_rr = MR_TP1_RR
            tp1_pct = int(MR_TP1_SPLIT * 100)
            tp2_pct = int(MR_TP2_SPLIT * 100)
        else:
            tp1_rr = CB_TP1_RR
            tp1_pct = int(CB_TP1_SPLIT * 100)
            tp2_pct = int(CB_TP2_SPLIT * 100)

        print(f"\n{'─' * 100}")
        print(f"TRADE #{idx + 1} | {trade['symbol']} | [{strat}] {trade['type']} | {emoji} {trade['result']} | {trade['pnl_r']:+.2f}R")
        print(f"{'─' * 100}")
        entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(trade['entry_time']) else "N/A"
        exit_time = trade['exit_time'].strftime('%H:%M') if pd.notna(trade['exit_time']) else "N/A"
        breakout_time = trade['breakout_time'].strftime('%H:%M') if pd.notna(trade['breakout_time']) else "N/A"
        print(f"  Session: {trade['session']} | Breakout: {breakout_time} | Entry: {entry_time} | Exit: {exit_time}")
        bo_dur = trade.get('breakout_duration_min')
        conf_count = trade.get('confirmation_count', 'N/A')
        if bo_dur is not None:
            print(f"  Breakout Duration: {bo_dur:.1f} min | Candles: {conf_count}")
        else:
            print(f"  Attente: {conf_count} bougies apres breakout")

        print(f"\n  SCENARIO: {scenario_label}")
        print(f"\n  PRIX:")
        print(f"    Entry:  {trade['entry']:.2f}")
        print(f"    SL:     {trade['sl']:.2f}")
        if pd.notna(trade.get('tp1')):
            tp1_status = "HIT" if pd.notna(trade.get('tp1_time')) else "not hit"
            print(f"    TP1:    {trade['tp1']:.2f} ({tp1_rr}R) {tp1_status}")
        tp2_status = "HIT" if pd.notna(trade.get('tp2_time')) else "not hit"
        tp_label = trade.get('tp_label', '?')
        print(f"    TP2:    {trade['tp']:.2f} ({tp_label}) {tp2_status}")
        print(f"\n  VP SESSION:")
        print(f"    VAH: {trade['vah_at_entry']:.2f} | POC: {trade['poc_at_entry']:.2f} | VAL: {trade['val_at_entry']:.2f}")
        print(f"    POC Strength: {trade['poc_strength']:.2f}x | Shape: {trade['vp_shape']}")
        if strat == 'CB':
            print(f"\n  STRUCTURAL LEVELS:")
            print(f"    TP target: {tp_label} = {trade['tp']:.2f}")
            prev_day_str = f"VAH={trade.get('pd_vah', 0):.2f} POC={trade.get('pd_poc', 0):.2f} VAL={trade.get('pd_val', 0):.2f}" if trade.get('pd_vah') else "N/A"
            prev_week_str = f"VAH={trade.get('pw_vah', 0):.2f} POC={trade.get('pw_poc', 0):.2f} VAL={trade.get('pw_val', 0):.2f}" if trade.get('pw_vah') else "N/A"
            print(f"    Prev Day:  {prev_day_str}")
            print(f"    Prev Week: {prev_week_str}")
        print(f"\n  RESULTAT:")
        print(f"    R:R: {trade['rr']:.2f} | Split: {tp1_pct}/{tp2_pct}")
        print(f"    PnL: {trade['pnl_r']:+.2f}R (${trade['pnl']:+.2f})")
        print(f"    Capital apres: ${trade['capital_after']:,.2f}")


# =============================================================================
# MAIN BACKTEST
# =============================================================================
def run_backtest():
    conn = get_db_connection()
    enabled_assets = [a for a in ASSETS if a.get('enabled', True)]
    if not enabled_assets:
        print("[ERROR] Aucun asset active.")
        return

    print("[DATA] Chargement des donnees...")
    t0 = time.time()

    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    assets_data = {}
    for asset in enabled_assets:
        df_candles, df_ticks = load_all_data(conn, asset)
        if not df_candles.empty:
            df_ticks['minute'] = df_ticks['time'].dt.floor('T')
            ticks_by_minute = df_ticks.groupby('minute').apply(
                lambda g: (g['price'].values, g['volume'].values, g['time'].values)
            ).to_dict()

            assets_data[asset['symbol']] = {
                'config': asset,
                'candles': df_candles,
                'ticks_by_minute': ticks_by_minute,
                'vp': IncrementalVolumeProfile(tick_size=asset['tick_size'], va_percent=asset['va_percent']),
                'structural': StructuralLevelsTracker(tick_size=asset['tick_size'], va_percent=asset['va_percent']),
                'state': "INSIDE",
                'active_trade': None,
                'current_session': None,
                'session_start_dt': None,
                # Breakout tracking
                'breakout_direction': None,
                'breakout_time': None,
                'breakout_level': None,
                'candles_since_breakout': 0,
                'wait_highs': [],
                'wait_lows': [],
                # Cooldown
                'last_loss_time': None,
                'last_loss_direction': None,
            }
            print(f"   + {asset['symbol']}: {len(df_candles):,} candles | {len(df_ticks):,} ticks")
    conn.close()
    print(f"   [TIME] Charge en {time.time() - t0:.2f}s")

    if not assets_data:
        print("[ERROR] Aucune donnee chargee.")
        return

    # Build unified DataFrame of ALL candles (no session filtering).
    # Structural levels are fed inline in the main loop (needs all candles).
    # Trade management also runs on all candles (SL/TP can be hit off-session).
    # The state machine is guarded by session checks (entries only during active sessions).
    all_candles = []
    for symbol, data in assets_data.items():
        df = data['candles'].copy()
        df['symbol'] = symbol
        all_candles.append(df)
    df_all_candles = pd.concat(all_candles).sort_values(['dt', 'symbol']).reset_index(drop=True)

    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    current_capital = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    max_dd_amount = 0.0
    max_dd_percent = 0.0
    all_trades = []

    # Filter counters
    total_mr_attempts = 0
    total_cb_attempts = 0
    filtered_mr_poc_strength = 0
    filtered_mr_rr = 0
    filtered_mr_duration = 0
    filtered_mr_poc_filter = 0
    filtered_cb_poc_strength = 0
    filtered_cb_no_target = 0

    # Period tracking
    current_day = None
    day_start_capital = INITIAL_CAPITAL
    day_trades = 0
    day_wins = 0
    day_pnl_r = 0.0
    day_high_water = INITIAL_CAPITAL
    day_max_dd = 0.0

    current_week = None
    week_start_capital = INITIAL_CAPITAL
    week_trades = 0
    week_wins = 0
    week_pnl_r = 0.0
    week_high_water = INITIAL_CAPITAL
    week_max_dd = 0.0

    current_month = None
    month_start_capital = INITIAL_CAPITAL
    month_trades = 0
    month_wins = 0
    month_pnl_r = 0.0
    month_high_water = INITIAL_CAPITAL
    month_max_dd = 0.0

    symbols_list = list(assets_data.keys())

    # ── HEADER ──
    strategies_active = []
    if ENABLE_MR: strategies_active.append("MR")
    if ENABLE_CB: strategies_active.append("CB")
    print(f"\n[BACKTEST] Combined MR + Confirmed Breakout | Strategies: {strategies_active}")
    print(f"[CAPITAL] ${INITIAL_CAPITAL:,.2f} | Risque: {RISK_PERCENT*100}%")
    print(f"[SHARED]  Wait candles: {WAIT_CANDLES} | Assets: {symbols_list}")
    if ENABLE_MR:
        mr_tp1_pct = int(MR_TP1_SPLIT * 100)
        mr_tp2_pct = int(MR_TP2_SPLIT * 100)
        print(f"[MR]      TP: {MR_TP_MODE} | RR >= {MR_MIN_RR} | SL offset: {MR_SL_OFFSET}")
        print(f"[MR]      Split: {mr_tp1_pct}/{mr_tp2_pct} | TP1 @ {MR_TP1_RR}R | Trailing: {'ON' if MR_USE_TRAILING else 'OFF'}")
        mr_dur_str = f"ON (max {MR_MAX_BREAKOUT_DURATION_MINUTES}min)" if MR_USE_BREAKOUT_DURATION_FILTER else "OFF"
        print(f"[MR]      POC strength >= {MR_MIN_POC_STRENGTH} | POC filter: {'ON' if MR_FILTER_ENTRY_VS_POC else 'OFF'} | Duration: {mr_dur_str}")
    if ENABLE_CB:
        cb_tp1_pct = int(CB_TP1_SPLIT * 100)
        cb_tp2_pct = int(CB_TP2_SPLIT * 100)
        print(f"[CB]      TP: structural levels | RR >= {CB_MIN_RR} | SL offset: {CB_SL_OFFSET}")
        print(f"[CB]      Split: {cb_tp1_pct}/{cb_tp2_pct} | TP1 @ {CB_TP1_RR}R | Trailing: {'ON' if CB_USE_TRAILING else 'OFF'}")
        print(f"[CB]      POC strength >= {CB_MIN_POC_STRENGTH} | Prev Day: {'ON' if CB_USE_PREV_DAY else 'OFF'} | Prev Week: {'ON' if CB_USE_PREV_WEEK else 'OFF'}")
    for sym in symbols_list:
        cfg = assets_data[sym]['config']
        sess = [s for s, v in cfg.get('sessions', {}).items() if v]
        day_names_short = {0: 'Lun', 1: 'Mar', 2: 'Mer', 3: 'Jeu', 4: 'Ven'}
        days_str = ', '.join(day_names_short.get(d, '?') for d in cfg.get('allowed_days', [0,1,2,3,4]))
        print(f"[CONFIG]  {sym}: Sessions={sess} | Jours=[{days_str}]")

    if DISPLAY_MODE != "NONE":
        print("=" * 140)
        if DISPLAY_MODE == "MONTHLY":
            print(f"{'MOIS':<10} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'PnL %':>8} | {'CAPITAL':>14} | {'MTH DD%':>8} | {'MAX DD%':>8}")
        elif DISPLAY_MODE == "WEEKLY":
            print(f"{'SEMAINE':<12} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'PnL %':>8} | {'CAPITAL':>14} | {'WK DD%':>8} | {'MAX DD%':>8}")
        elif DISPLAY_MODE == "DAILY":
            print(f"{'DATE':<12} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'PnL %':>8} | {'CAPITAL':>14} | {'DAY DD%':>8} | {'MAX DD%':>8}")
        print("-" * 140)

    t_start = time.time()

    for row in df_all_candles.itertuples():
        symbol = row.symbol
        ad = assets_data[symbol]
        config = ad['config']
        current_minute = row.dt.floor('T')
        row_day = row.dt.date()
        row_week_start = (row.dt - timedelta(days=row.dt.weekday())).date()
        row_week = row_week_start.strftime('%Y-%m-%d')
        row_month = row.dt.strftime('%Y-%m')

        # ── Period tracking ──
        if current_day is not None and row_day != current_day:
            if DISPLAY_MODE == "DAILY" and day_trades > 0:
                dwr = (day_wins / day_trades * 100) if day_trades > 0 else 0
                ddd = (day_max_dd / day_high_water * 100) if day_high_water > 0 else 0
                dpnl_pct = ((current_capital - day_start_capital) / day_start_capital * 100) if day_start_capital > 0 else 0
                print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades-day_wins:>4} | {dwr:>5.1f}% | {day_pnl_r:>+7.1f}R | {dpnl_pct:>+7.2f}% | ${current_capital:>13,.2f} | {ddd:>7.2f}% | {max_dd_percent:>7.2f}%")
            day_start_capital = current_capital; day_trades = 0; day_wins = 0; day_pnl_r = 0.0; day_high_water = current_capital; day_max_dd = 0.0

        if current_week is not None and row_week != current_week:
            if DISPLAY_MODE == "WEEKLY" and week_trades > 0:
                wwr = (week_wins / week_trades * 100) if week_trades > 0 else 0
                wdd = (week_max_dd / week_high_water * 100) if week_high_water > 0 else 0
                wpnl_pct = ((current_capital - week_start_capital) / week_start_capital * 100) if week_start_capital > 0 else 0
                print(f"{current_week:<12} | {week_trades:>6} | {week_wins:>4} | {week_trades-week_wins:>4} | {wwr:>5.1f}% | {week_pnl_r:>+7.1f}R | {wpnl_pct:>+7.2f}% | ${current_capital:>13,.2f} | {wdd:>7.2f}% | {max_dd_percent:>7.2f}%")
            week_start_capital = current_capital; week_trades = 0; week_wins = 0; week_pnl_r = 0.0; week_high_water = current_capital; week_max_dd = 0.0

        if current_month is not None and row_month != current_month:
            if DISPLAY_MODE == "MONTHLY" and month_trades > 0:
                mwr = (month_wins / month_trades * 100) if month_trades > 0 else 0
                mdd = (month_max_dd / month_high_water * 100) if month_high_water > 0 else 0
                mpnl_pct = ((current_capital - month_start_capital) / month_start_capital * 100) if month_start_capital > 0 else 0
                print(f"{current_month:<10} | {month_trades:>6} | {month_wins:>4} | {month_trades-month_wins:>4} | {mwr:>5.1f}% | {month_pnl_r:>+7.1f}R | {mpnl_pct:>+7.2f}% | ${current_capital:>13,.2f} | {mdd:>7.2f}% | {max_dd_percent:>7.2f}%")
            month_start_capital = current_capital; month_trades = 0; month_wins = 0; month_pnl_r = 0.0; month_high_water = current_capital; month_max_dd = 0.0

        current_day = row_day
        current_week = row_week
        current_month = row_month

        # =================================================================
        # STEP 1: STRUCTURAL LEVELS — feed from ALL candles (no session filter)
        # =================================================================
        if current_minute in ad['ticks_by_minute']:
            prices_struct, volumes_struct, _ = ad['ticks_by_minute'][current_minute]
            ad['structural'].update(row.dt, prices_struct, volumes_struct)

        # Skip warmup period (structural levels need 14 days of data)
        if row.dt < requested_start:
            continue

        # =================================================================
        # STEP 2: SESSION VP (only during active sessions)
        # =================================================================
        curr_sess = get_session(row.dt)
        asset_sessions = config.get('sessions', {})

        if RESET_VP_PER_SESSION:
            session_start = is_session_start(row.dt)
            if session_start and asset_sessions.get(session_start, False):
                ad['vp'].reset()
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0
                ad['wait_highs'] = []
                ad['wait_lows'] = []
                ad['current_session'] = session_start
                ad['session_start_dt'] = get_session_start_time(session_start, row.dt)

        if asset_sessions.get(curr_sess, False):
            if current_minute in ad['ticks_by_minute']:
                prices, volumes, timestamps = ad['ticks_by_minute'][current_minute]
                session_start_dt = ad.get('session_start_dt')
                if session_start_dt is not None:
                    session_start_np = np.datetime64(session_start_dt)
                    mask = timestamps >= session_start_np
                    if mask.any():
                        ad['vp'].add_ticks(prices[mask], volumes[mask])
                else:
                    ad['vp'].add_ticks(prices, volumes)

        # =================================================================
        # STEP 3: ACTIVE TRADE MANAGEMENT
        # =================================================================
        active_trade = ad['active_trade']
        if active_trade:
            res = None
            strat = active_trade['strategy']
            use_trailing = MR_USE_TRAILING if strat == 'MR' else CB_USE_TRAILING
            tp1_rr = MR_TP1_RR if strat == 'MR' else CB_TP1_RR
            tp1_split = MR_TP1_SPLIT if strat == 'MR' else CB_TP1_SPLIT
            tp2_split = MR_TP2_SPLIT if strat == 'MR' else CB_TP2_SPLIT

            if active_trade['type'] == 'LONG':
                if row.low <= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    if use_trailing and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] + (active_trade['risk'] * tp1_rr)
                        if row.high >= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = tp1_rr * tp1_split
                            active_trade['tp1_time'] = row.dt
                    if row.high >= active_trade['tp']:
                        res = "WIN"
                        active_trade['exit_time'] = row.dt
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['tp2_time'] = row.dt

            elif active_trade['type'] == 'SHORT':
                if row.high >= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    if use_trailing and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] - (active_trade['risk'] * tp1_rr)
                        if row.low <= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = tp1_rr * tp1_split
                            active_trade['tp1_time'] = row.dt
                    if row.low <= active_trade['tp']:
                        res = "WIN"
                        active_trade['exit_time'] = row.dt
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['tp2_time'] = row.dt

            if res:
                risk_amount = current_capital * RISK_PERCENT
                trade_rr = active_trade['rr']
                partial_pnl_r = active_trade.get('partial_pnl_r', 0)
                if res == "WIN":
                    if use_trailing and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r + (trade_rr * tp2_split)
                    else:
                        pnl_r = trade_rr
                    pnl = risk_amount * pnl_r
                    day_wins += 1
                else:
                    if use_trailing and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r
                        pnl = risk_amount * pnl_r
                        res = "BE"
                        active_trade['exit_price'] = active_trade['entry']
                        day_wins += 1
                    else:
                        pnl = -risk_amount
                        pnl_r = -1.0
                        if USE_COOLDOWN_AFTER_LOSS:
                            ad['last_loss_time'] = row.dt
                            ad['last_loss_direction'] = active_trade['type']

                current_capital += pnl
                day_trades += 1; day_pnl_r += pnl_r
                week_trades += 1; week_pnl_r += pnl_r
                month_trades += 1; month_pnl_r += pnl_r
                if res in ["WIN", "BE"]:
                    week_wins += 1; month_wins += 1

                if current_capital > high_water_mark: high_water_mark = current_capital
                if current_capital > day_high_water: day_high_water = current_capital
                if current_capital > week_high_water: week_high_water = current_capital
                if current_capital > month_high_water: month_high_water = current_capital

                current_dd = high_water_mark - current_capital
                current_dd_pct = (current_dd / high_water_mark * 100) if high_water_mark > 0 else 0
                if current_dd > max_dd_amount: max_dd_amount = current_dd
                if current_dd_pct > max_dd_percent: max_dd_percent = current_dd_pct

                dd = day_high_water - current_capital
                if dd > day_max_dd: day_max_dd = dd
                dd = week_high_water - current_capital
                if dd > week_max_dd: week_max_dd = dd
                dd = month_high_water - current_capital
                if dd > month_max_dd: month_max_dd = dd

                all_trades.append({
                    'symbol': symbol, 'date': row.dt,
                    'entry_hour': active_trade.get('entry_time').hour if active_trade.get('entry_time') else row.dt.hour,
                    'session': active_trade['session_at_open'], 'type': active_trade['type'],
                    'strategy': active_trade['strategy'],
                    'breakout_time': active_trade.get('breakout_time'),
                    'breakout_duration_min': active_trade.get('breakout_duration_min'),
                    'confirmation_count': active_trade.get('wait_candles'),
                    'poc_strength': active_trade.get('poc_strength'), 'vp_shape': active_trade.get('vp_shape'),
                    'entry_time': active_trade.get('entry_time'), 'entry': active_trade['entry'],
                    'sl': active_trade['original_sl'], 'tp1': active_trade.get('tp1'),
                    'tp1_time': active_trade.get('tp1_time'), 'tp': active_trade['tp'],
                    'tp_label': active_trade.get('tp_label'),
                    'tp2_time': active_trade.get('tp2_time'), 'exit_time': active_trade.get('exit_time'),
                    'exit_price': active_trade.get('exit_price'),
                    'vah_at_entry': active_trade.get('vah_at_entry'),
                    'val_at_entry': active_trade.get('val_at_entry'),
                    'poc_at_entry': active_trade.get('poc_at_entry'),
                    'pd_vah': active_trade.get('pd_vah'), 'pd_val': active_trade.get('pd_val'), 'pd_poc': active_trade.get('pd_poc'),
                    'pw_vah': active_trade.get('pw_vah'), 'pw_val': active_trade.get('pw_val'), 'pw_poc': active_trade.get('pw_poc'),
                    'rr': active_trade['rr'], 'result': res, 'pnl': pnl, 'pnl_r': pnl_r,
                    'capital_after': current_capital, 'high_water_mark': high_water_mark,
                    'drawdown': high_water_mark - current_capital
                })
                ad['active_trade'] = None
            else:
                continue

        # =================================================================
        # STEP 4: STATE MACHINE — Combined MR + CB
        # =================================================================
        if not asset_sessions.get(curr_sess, False):
            ad['state'] = "INSIDE"
            ad['breakout_direction'] = None
            ad['candles_since_breakout'] = 0
            continue

        poc, vah, val = ad['vp'].get_levels()
        if poc is None:
            continue

        poc_strength = ad['vp'].get_poc_strength()
        vp_shape = ad['vp'].get_profile_shape()
        close, high, low = row.close, row.high, row.low
        state = ad['state']

        if state == "INSIDE":
            if close > vah:
                ad['state'] = "BREAKOUT_UP"
                ad['breakout_direction'] = "UP"
                ad['breakout_time'] = row.dt
                ad['breakout_level'] = vah
                ad['candles_since_breakout'] = 1
                ad['wait_highs'] = [high]
                ad['wait_lows'] = [low]
            elif close < val:
                ad['state'] = "BREAKOUT_DOWN"
                ad['breakout_direction'] = "DOWN"
                ad['breakout_time'] = row.dt
                ad['breakout_level'] = val
                ad['candles_since_breakout'] = 1
                ad['wait_highs'] = [high]
                ad['wait_lows'] = [low]

        elif state in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
            ad['candles_since_breakout'] += 1
            ad['wait_highs'].append(high)
            ad['wait_lows'].append(low)

            breakout_dir = ad['breakout_direction']
            reintegrated = False
            confirmed_outside = False

            # Check reintegration
            # MR uses strict (close < vah) to match original MR behavior
            # CB uses non-strict (close <= vah) to match original CB behavior
            if ENABLE_MR:
                # Strict: must close INSIDE VA, not just at boundary
                if breakout_dir == "UP" and close < vah:
                    reintegrated = True
                elif breakout_dir == "DOWN" and close > val:
                    reintegrated = True
            else:
                # Non-strict: close at boundary counts as reintegration (CB original)
                if breakout_dir == "UP" and close <= vah:
                    reintegrated = True
                elif breakout_dir == "DOWN" and close >= val:
                    reintegrated = True

            # Check confirmed breakout (waited enough candles AND still outside)
            if not reintegrated and ad['candles_since_breakout'] >= WAIT_CANDLES:
                confirmed_outside = True

            trade_attempted = False

            # ── REINTEGRATION → attempt MR ──
            if reintegrated and ENABLE_MR:
                total_mr_attempts += 1
                # MR direction: opposite to breakout
                if breakout_dir == "UP":
                    direction = 'SHORT'
                else:
                    direction = 'LONG'

                # Common filters
                can_direction = (direction == 'LONG' and config['allow_long']) or (direction == 'SHORT' and config['allow_short'])
                can_time = can_trade_now(row.dt, curr_sess)
                day_ok = row.dt.weekday() in config.get('allowed_days', [0,1,2,3,4])
                hour_ok = row.dt.hour not in MR_EXCLUDED_HOURS

                # MR-specific filters
                mr_poc_ok = True
                if MR_USE_VP_STRUCTURE_FILTER:
                    if poc_strength is None or poc_strength < MR_MIN_POC_STRENGTH:
                        mr_poc_ok = False
                        filtered_mr_poc_strength += 1

                mr_duration_ok = True
                if MR_USE_BREAKOUT_DURATION_FILTER:
                    breakout_duration_min = (row.dt - ad['breakout_time']).total_seconds() / 60.0
                    if breakout_duration_min >= MR_MAX_BREAKOUT_DURATION_MINUTES:
                        mr_duration_ok = False
                        filtered_mr_duration += 1

                shape_ok = True
                if USE_VP_SHAPE_FILTER and vp_shape in EXCLUDED_VP_SHAPES:
                    shape_ok = False

                cooldown_ok = True
                if USE_COOLDOWN_AFTER_LOSS and ad['last_loss_time'] is not None:
                    if ad['last_loss_direction'] == direction:
                        mins = (row.dt - ad['last_loss_time']).total_seconds() / 60.0
                        if mins < COOLDOWN_AFTER_LOSS_MINUTES:
                            cooldown_ok = False

                if mr_poc_ok and mr_duration_ok and shape_ok and cooldown_ok and can_direction and can_time and day_ok and hour_ok:
                    # SL: swing extreme of breakout candles + offset (per-asset or global)
                    mr_sl_offset = config.get('sl_offset', MR_SL_OFFSET)
                    if direction == 'SHORT':
                        swing_high = max(ad['wait_highs'])
                        sl = swing_high + mr_sl_offset
                        risk = sl - close
                    else:
                        swing_low = min(ad['wait_lows'])
                        sl = swing_low - mr_sl_offset
                        risk = close - sl

                    if risk > 0:
                        # TP: POC of session VP
                        tp = poc

                        if direction == 'SHORT':
                            actual_rr = (close - tp) / risk if risk > 0 else 0
                        else:
                            actual_rr = (tp - close) / risk if risk > 0 else 0

                        # MR POC filter: entry must be on correct side of POC
                        poc_filter_ok = True
                        if MR_FILTER_ENTRY_VS_POC:
                            if direction == 'SHORT' and close < poc:
                                poc_filter_ok = False
                                filtered_mr_poc_filter += 1
                            elif direction == 'LONG' and close > poc:
                                poc_filter_ok = False
                                filtered_mr_poc_filter += 1

                        rr_ok = actual_rr >= MR_MIN_RR and actual_rr <= 30
                        if not rr_ok and actual_rr < MR_MIN_RR:
                            filtered_mr_rr += 1

                        # TP must be within VA
                        tp_in_va = True
                        if direction == 'SHORT' and tp < val:
                            tp_in_va = False
                        if direction == 'LONG' and tp > vah:
                            tp_in_va = False

                        if rr_ok and poc_filter_ok and tp_in_va:
                            tp1_price = None
                            if MR_USE_TRAILING:
                                if direction == 'LONG':
                                    tp1_price = close + (risk * MR_TP1_RR)
                                else:
                                    tp1_price = close - (risk * MR_TP1_RR)

                            pd_levels = ad['structural'].prev_day
                            pw_levels = ad['structural'].prev_week

                            mr_bo_duration = (row.dt - ad['breakout_time']).total_seconds() / 60.0
                            ad['active_trade'] = {
                                'type': direction, 'strategy': 'MR',
                                'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
                                'tp': tp, 'tp_label': 'POC', 'tp1': tp1_price, 'rr': actual_rr,
                                'session_at_open': curr_sess,
                                'partial_closed': False, 'partial_pnl_r': 0,
                                'breakout_time': ad['breakout_time'], 'entry_time': row.dt,
                                'wait_candles': ad['candles_since_breakout'],
                                'breakout_duration_min': mr_bo_duration,
                                'poc_strength': poc_strength, 'vp_shape': vp_shape,
                                'tp1_time': None, 'tp2_time': None, 'exit_time': None, 'exit_price': None,
                                'vah_at_entry': vah, 'val_at_entry': val, 'poc_at_entry': poc,
                                'pd_vah': pd_levels['vah'], 'pd_val': pd_levels['val'], 'pd_poc': pd_levels['poc'],
                                'pw_vah': pw_levels['vah'], 'pw_val': pw_levels['val'], 'pw_poc': pw_levels['poc'],
                            }
                            trade_attempted = True

                # Reset state after MR attempt (successful or not)
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0
                ad['wait_highs'] = []
                ad['wait_lows'] = []

            # ── CONFIRMED OUTSIDE → attempt CB ──
            elif confirmed_outside and ENABLE_CB:
                total_cb_attempts += 1
                # CB direction: same as breakout
                if breakout_dir == "UP":
                    direction = 'LONG'
                else:
                    direction = 'SHORT'

                # Common filters
                can_direction = (direction == 'LONG' and config['allow_long']) or (direction == 'SHORT' and config['allow_short'])
                can_time = can_trade_now(row.dt, curr_sess)
                day_ok = row.dt.weekday() in config.get('allowed_days', [0,1,2,3,4])
                hour_ok = row.dt.hour not in CB_EXCLUDED_HOURS

                cb_poc_ok = True
                if CB_USE_VP_STRUCTURE_FILTER:
                    if poc_strength is None or poc_strength < CB_MIN_POC_STRENGTH:
                        cb_poc_ok = False
                        filtered_cb_poc_strength += 1

                shape_ok = True
                if USE_VP_SHAPE_FILTER and vp_shape in EXCLUDED_VP_SHAPES:
                    shape_ok = False

                cooldown_ok = True
                if USE_COOLDOWN_AFTER_LOSS and ad['last_loss_time'] is not None:
                    if ad['last_loss_direction'] == direction:
                        mins = (row.dt - ad['last_loss_time']).total_seconds() / 60.0
                        if mins < COOLDOWN_AFTER_LOSS_MINUTES:
                            cooldown_ok = False

                if cb_poc_ok and shape_ok and cooldown_ok and can_direction and can_time and day_ok and hour_ok:
                    # SL: swing low/high of wait candles + offset
                    if direction == 'LONG':
                        swing_low = min(ad['wait_lows'])
                        sl = swing_low - CB_SL_OFFSET
                        risk = close - sl
                    else:
                        swing_high = max(ad['wait_highs'])
                        sl = swing_high + CB_SL_OFFSET
                        risk = sl - close

                    if risk > 0:
                        # TP: structural level (prev day/week)
                        targets = ad['structural'].get_target_levels(direction, close)

                        tp = None
                        tp_label = None
                        for label, price in targets:
                            if direction == 'LONG':
                                rr = (price - close) / risk
                            else:
                                rr = (close - price) / risk
                            if rr >= CB_MIN_RR and rr <= 30:
                                tp = price
                                tp_label = label
                                break

                        if tp is None:
                            filtered_cb_no_target += 1
                            # Fallback: fixed RR
                            if direction == 'LONG':
                                tp = close + (risk * CB_MIN_RR)
                            else:
                                tp = close - (risk * CB_MIN_RR)
                            tp_label = f"FIXED_{CB_MIN_RR}R"
                            actual_rr = CB_MIN_RR
                        else:
                            if direction == 'LONG':
                                actual_rr = (tp - close) / risk
                            else:
                                actual_rr = (close - tp) / risk

                        tp1_price = None
                        if CB_USE_TRAILING:
                            if direction == 'LONG':
                                tp1_price = close + (risk * CB_TP1_RR)
                            else:
                                tp1_price = close - (risk * CB_TP1_RR)

                        pd_levels = ad['structural'].prev_day
                        pw_levels = ad['structural'].prev_week

                        ad['active_trade'] = {
                            'type': direction, 'strategy': 'CB',
                            'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
                            'tp': tp, 'tp_label': tp_label, 'tp1': tp1_price, 'rr': actual_rr,
                            'session_at_open': curr_sess,
                            'partial_closed': False, 'partial_pnl_r': 0,
                            'breakout_time': ad['breakout_time'], 'entry_time': row.dt,
                            'wait_candles': ad['candles_since_breakout'],
                            'poc_strength': poc_strength, 'vp_shape': vp_shape,
                            'tp1_time': None, 'tp2_time': None, 'exit_time': None, 'exit_price': None,
                            'vah_at_entry': vah, 'val_at_entry': val, 'poc_at_entry': poc,
                            'pd_vah': pd_levels['vah'], 'pd_val': pd_levels['val'], 'pd_poc': pd_levels['poc'],
                            'pw_vah': pw_levels['vah'], 'pw_val': pw_levels['val'], 'pw_poc': pw_levels['poc'],
                        }
                        trade_attempted = True

                # Reset state after CB attempt
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0
                ad['wait_highs'] = []
                ad['wait_lows'] = []

            elif reintegrated and not ENABLE_MR:
                # MR disabled, reintegration just resets
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0
                ad['wait_highs'] = []
                ad['wait_lows'] = []

            elif confirmed_outside and not ENABLE_CB:
                # CB disabled, sustained breakout: stay in BREAKOUT state
                # (don't reset to INSIDE, keep waiting for reintegration like original MR)
                pass

            # If not reintegrated and not yet enough candles, stay in BREAKOUT state

    # ── Last period ──
    if DISPLAY_MODE == "DAILY" and day_trades > 0:
        dwr = (day_wins/day_trades*100) if day_trades>0 else 0; ddd = (day_max_dd/day_high_water*100) if day_high_water>0 else 0; dpnl_pct = ((current_capital-day_start_capital)/day_start_capital*100) if day_start_capital>0 else 0
        print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades-day_wins:>4} | {dwr:>5.1f}% | {day_pnl_r:>+7.1f}R | {dpnl_pct:>+7.2f}% | ${current_capital:>13,.2f} | {ddd:>7.2f}% | {max_dd_percent:>7.2f}%")
    if DISPLAY_MODE == "WEEKLY" and week_trades > 0:
        wwr = (week_wins/week_trades*100) if week_trades>0 else 0; wdd = (week_max_dd/week_high_water*100) if week_high_water>0 else 0; wpnl_pct = ((current_capital-week_start_capital)/week_start_capital*100) if week_start_capital>0 else 0
        print(f"{current_week:<12} | {week_trades:>6} | {week_wins:>4} | {week_trades-week_wins:>4} | {wwr:>5.1f}% | {week_pnl_r:>+7.1f}R | {wpnl_pct:>+7.2f}% | ${current_capital:>13,.2f} | {wdd:>7.2f}% | {max_dd_percent:>7.2f}%")
    if DISPLAY_MODE == "MONTHLY" and month_trades > 0:
        mwr = (month_wins/month_trades*100) if month_trades>0 else 0; mdd = (month_max_dd/month_high_water*100) if month_high_water>0 else 0; mpnl_pct = ((current_capital-month_start_capital)/month_start_capital*100) if month_start_capital>0 else 0
        print(f"{current_month:<10} | {month_trades:>6} | {month_wins:>4} | {month_trades-month_wins:>4} | {mwr:>5.1f}% | {month_pnl_r:>+7.1f}R | {mpnl_pct:>+7.2f}% | ${current_capital:>13,.2f} | {mdd:>7.2f}% | {max_dd_percent:>7.2f}%")

    elapsed = time.time() - t_start

    # =================================================================
    # FINAL REPORT
    # =================================================================
    print("\n" + "=" * 120)
    print("[RAPPORT FINAL] Combined MR + Confirmed Breakout")
    print("=" * 120)

    if not all_trades:
        print("[ERROR] Aucun trade execute.")
        return

    df_trades = pd.DataFrame(all_trades)
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    be_trades = len(df_trades[df_trades['result'] == 'BE'])
    losses = len(df_trades[df_trades['result'] == 'LOSS'])
    win_rate = (wins + be_trades) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = df_trades['pnl'].sum()
    total_pnl_r = df_trades['pnl_r'].sum()
    avg_pnl_r = df_trades['pnl_r'].mean()
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    positive_trades = df_trades[df_trades['result'].isin(['WIN', 'BE'])]
    avg_win = positive_trades['pnl_r'].mean() if len(positive_trades) > 0 else 0
    avg_loss_val = abs(df_trades[df_trades['result'] == 'LOSS']['pnl_r'].mean()) if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss_val)
    recovery_factor = total_pnl / max_dd_amount if max_dd_amount > 0 else float('inf')
    trading_days = df_trades['date'].dt.date.nunique()
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0

    df_trades['is_loss'] = df_trades['result'] == 'LOSS'
    df_trades['streak'] = (df_trades['is_loss'] != df_trades['is_loss'].shift()).cumsum()
    losing_streaks = df_trades[df_trades['is_loss']].groupby('streak').size()
    winning_streaks = df_trades[~df_trades['is_loss']].groupby('streak').size()
    max_losing_streak = losing_streaks.max() if len(losing_streaks) > 0 else 0
    max_winning_streak = winning_streaks.max() if len(winning_streaks) > 0 else 0

    print(f"\n{'─' * 60}")
    print("PERFORMANCE GLOBALE")
    print(f"{'─' * 60}")
    print(f"  Capital Initial:      ${INITIAL_CAPITAL:>14,.2f}")
    print(f"  Capital Final:        ${current_capital:>14,.2f}")
    print(f"  Profit/Perte:         ${total_pnl:>+14,.2f} ({total_pnl/INITIAL_CAPITAL*100:+.2f}%)")
    print(f"  Gain Total:           {total_pnl_r:>+14.1f} R")
    print(f"  Max Drawdown:         {max_dd_percent:>13.2f}%")
    print(f"  Recovery Factor:      {recovery_factor:>14.2f}")

    print(f"\n{'─' * 60}")
    print("STATISTIQUES DE TRADING")
    print(f"{'─' * 60}")
    print(f"  Trades Total:         {total_trades:>14}")
    if be_trades > 0:
        print(f"  Wins / BE / Losses:   {wins:>5} / {be_trades:>3} / {losses:<5}")
    else:
        print(f"  Wins / Losses:        {wins:>6} / {losses:<6}")
    print(f"  Win Rate:             {win_rate:>13.2f}%")
    print(f"  Profit Factor:        {profit_factor:>14.2f}")
    print(f"  Expectancy:           {expectancy:>+13.2f} R")
    print(f"  Avg PnL/Trade:        {avg_pnl_r:>+13.2f} R")
    avg_rr_winners = df_trades[df_trades['result'] == 'WIN']['rr'].mean() if wins > 0 else 0
    print(f"  Avg R:R (winners):    {avg_rr_winners:>14.2f}")

    # Exit scenarios
    print(f"\n{'─' * 60}")
    print("SCENARIOS DE SORTIE")
    print(f"{'─' * 60}")
    tp1_tp2 = len(df_trades[(df_trades['result']=='WIN') & (df_trades['tp1_time'].notna())])
    tp2_direct = len(df_trades[(df_trades['result']=='WIN') & (df_trades['tp1_time'].isna())])
    sl_direct = len(df_trades[(df_trades['result']=='LOSS') & (df_trades['tp1_time'].isna())])
    print(f"  TP1 -> TP2 (full win):  {tp1_tp2:>5}")
    print(f"  TP1 -> BE:              {be_trades:>5}")
    print(f"  SL direct:              {sl_direct:>5}")
    if tp2_direct > 0:
        print(f"  TP2 direct:             {tp2_direct:>5}")

    # ── STRATEGY BREAKDOWN ──
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR STRATEGIE")
    print(f"{'─' * 60}")
    print(f"{'STRAT':<6} | {'TRADES':>7} | {'WIN':>5} | {'BE':>4} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'Avg R':>7} | {'PF':>6}")
    print("-" * 80)
    for strat_name in ['MR', 'CB']:
        st = df_trades[df_trades['strategy'] == strat_name]
        if len(st) == 0:
            continue
        s_total = len(st)
        s_wins = len(st[st['result'] == 'WIN'])
        s_be = len(st[st['result'] == 'BE'])
        s_losses = len(st[st['result'] == 'LOSS'])
        s_wr = (s_wins + s_be) / s_total * 100
        s_pnl = st['pnl_r'].sum()
        s_avg = st['pnl_r'].mean()
        s_gp = st[st['pnl'] > 0]['pnl'].sum()
        s_gl = abs(st[st['pnl'] < 0]['pnl'].sum())
        s_pf = s_gp / s_gl if s_gl > 0 else float('inf')
        pf_str = f"{s_pf:.2f}" if s_pf != float('inf') else "inf"
        print(f"{strat_name:<6} | {s_total:>7} | {s_wins:>5} | {s_be:>4} | {s_losses:>5} | {s_wr:>6.1f}% | {s_pnl:>+8.1f}R | {s_avg:>+6.2f}R | {pf_str:>6}")

    # Filters
    print(f"\n{'─' * 60}")
    print("FILTRES APPLIQUES")
    print(f"{'─' * 60}")
    if ENABLE_MR:
        print(f"  [MR] Tentatives:        {total_mr_attempts:>10}")
        print(f"  [MR] POC strength:      {filtered_mr_poc_strength:>10}")
        print(f"  [MR] Duration:          {filtered_mr_duration:>10}")
        print(f"  [MR] POC filter:        {filtered_mr_poc_filter:>10}")
        print(f"  [MR] R:R too low:       {filtered_mr_rr:>10}")
    if ENABLE_CB:
        print(f"  [CB] Tentatives:        {total_cb_attempts:>10}")
        print(f"  [CB] POC strength:      {filtered_cb_poc_strength:>10}")
        print(f"  [CB] No target:         {filtered_cb_no_target:>10}")

    # Performance by TP target (CB trades)
    if 'tp_label' in df_trades.columns:
        cb_trades = df_trades[df_trades['strategy'] == 'CB']
        if len(cb_trades) > 0:
            print(f"\n{'─' * 60}")
            print("PERFORMANCE PAR NIVEAU CIBLE (CB)")
            print(f"{'─' * 60}")
            for label in ['PD_VAH', 'PD_POC', 'PD_VAL', 'PW_VAH', 'PW_POC', 'PW_VAL']:
                lt = cb_trades[cb_trades['tp_label'] == label]
                if len(lt) > 0:
                    lw = len(lt[lt['result'].isin(['WIN', 'BE'])])
                    lwr = lw / len(lt) * 100
                    lpnl = lt['pnl_r'].sum()
                    lavg = lt['pnl_r'].mean()
                    lgp = lt[lt['pnl'] > 0]['pnl'].sum()
                    lgl = abs(lt[lt['pnl'] < 0]['pnl'].sum())
                    lpf = lgp / lgl if lgl > 0 else float('inf')
                    pf_str = f"{lpf:.2f}" if lpf != float('inf') else "inf"
                    label_display = label.replace('PD_', 'Prev Day ').replace('PW_', 'Prev Week ')
                    print(f"  {label_display:<20} | {len(lt):>4} trades | WR: {lwr:>5.1f}% | PnL: {lpnl:>+8.1f}R | Avg: {lavg:>+5.2f}R | PF: {pf_str}")

    # Breakdown session
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR SESSION")
    print(f"{'─' * 60}")
    print(f"{'SESSION':<10} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6}")
    print("-" * 70)
    for session in ["TOKYO", "LONDON", "NY"]:
        st = df_trades[df_trades['session'] == session]
        if len(st) == 0: continue
        sw = len(st[st['result'].isin(['WIN', 'BE'])])
        sl_count = len(st[st['result'] == 'LOSS'])
        swr = sw / len(st) * 100
        spnl = st['pnl_r'].sum()
        sgp = st[st['pnl'] > 0]['pnl'].sum()
        sgl = abs(st[st['pnl'] < 0]['pnl'].sum())
        spf = sgp / sgl if sgl > 0 else float('inf')
        pf_str = f"{spf:.2f}" if spf != float('inf') else "inf"
        print(f"{session:<10} | {len(st):>7} | {sw:>5} | {sl_count:>5} | {swr:>6.1f}% | {spnl:>+8.1f}R | {pf_str:>6}")

    # Breakdown direction
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR DIRECTION")
    print(f"{'─' * 60}")
    for d in ["LONG", "SHORT"]:
        dt_dir = df_trades[df_trades['type'] == d]
        if len(dt_dir) == 0: continue
        dw = len(dt_dir[dt_dir['result'].isin(['WIN', 'BE'])])
        dl = len(dt_dir[dt_dir['result'] == 'LOSS'])
        dwr_val = dw / len(dt_dir) * 100
        dpnl = dt_dir['pnl_r'].sum()
        dgp = dt_dir[dt_dir['pnl'] > 0]['pnl'].sum()
        dgl = abs(dt_dir[dt_dir['pnl'] < 0]['pnl'].sum())
        dpf = dgp / dgl if dgl > 0 else float('inf')
        pf_str = f"{dpf:.2f}" if dpf != float('inf') else "inf"
        print(f"  {d:<8} | {len(dt_dir):>7} trades | {dw:>5} W | {dl:>5} L | {dwr_val:>5.1f}% | {dpnl:>+8.1f}R | PF {pf_str}")

    # Breakdown day
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR JOUR")
    print(f"{'─' * 60}")
    day_names = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi'}
    df_trades['day_of_week'] = df_trades['date'].dt.dayofweek
    for dow in range(5):
        dt_dow = df_trades[df_trades['day_of_week'] == dow]
        if len(dt_dow) == 0: continue
        dw = len(dt_dow[dt_dow['result'].isin(['WIN', 'BE'])])
        dl = len(dt_dow[dt_dow['result'] == 'LOSS'])
        dwr_val = dw / len(dt_dow) * 100
        dpnl = dt_dow['pnl_r'].sum()
        davg = dt_dow['pnl_r'].mean()
        print(f"  {day_names[dow]:<12} | {len(dt_dow):>4} trades | WR: {dwr_val:>5.1f}% | PnL: {dpnl:>+8.1f}R | Avg: {davg:>+5.2f}R")

    # Breakdown hour
    if 'entry_hour' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR HEURE (UTC)")
        print(f"{'─' * 60}")
        for hour in range(24):
            ht = df_trades[df_trades['entry_hour'] == hour]
            if len(ht) == 0: continue
            hw = len(ht[ht['result'].isin(['WIN', 'BE'])])
            hwr = hw / len(ht) * 100
            hpnl = ht['pnl_r'].sum()
            havg = ht['pnl_r'].mean()
            if 0 <= hour < 4: tag = "TKY"
            elif 8 <= hour < 14: tag = "LDN"
            elif 14 <= hour < 21: tag = "NYC"
            else: tag = "OFF"
            print(f"  {tag} {hour:02d}:00 | {len(ht):>4} trades | WR: {hwr:>5.1f}% | PnL: {hpnl:>+8.1f}R | Avg: {havg:>+5.2f}R")

    print(f"\n{'─' * 60}")
    print("STATISTIQUES TEMPORELLES")
    print(f"{'─' * 60}")
    print(f"  Jours de Trading:     {trading_days:>14}")
    print(f"  Trades/Jour (avg):    {avg_trades_per_day:>14.1f}")
    print(f"  Max Serie Gagnante:   {max_winning_streak:>14}")
    print(f"  Max Serie Perdante:   {max_losing_streak:>14}")
    print(f"  Temps d'execution:    {elapsed:>13.2f}s")

    if SHOW_LAST_TRADES:
        display_last_trades(df_trades, LAST_TRADES_COUNT)

    print("=" * 120)


if __name__ == "__main__":
    run_backtest()
