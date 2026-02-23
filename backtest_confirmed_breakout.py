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
# CONFIGURATION
# =============================================================================

START_DATE_STR = "2025-01-01 00:00:00"
INITIAL_CAPITAL = 1000
RISK_PERCENT = 0.01

MIN_RR = 2.0

USE_TRAILING = True
TP1_RR = 1.0
TP1_SPLIT = 0.3
TP2_SPLIT = 0.7

# =============================================================================
# BREAKOUT CONFIRMATION
# =============================================================================
WAIT_CANDLES = 3                  # Attendre N bougies apres le breakout. Si toujours dehors -> entree

# =============================================================================
# NIVEAUX STRUCTURELS POUR TP
# =============================================================================
USE_PREV_DAY_LEVELS = True        # Utiliser prev day VAH/VAL/POC comme TP
USE_PREV_WEEK_LEVELS = True       # Utiliser prev week VAH/VAL/POC comme TP
SL_OFFSET_POINTS = 1.0            # Offset SL au-dela du swing des bougies de confirmation

# =============================================================================
# FILTRES
# =============================================================================
USE_VP_STRUCTURE_FILTER = True
MIN_POC_STRENGTH = 3.0
USE_VP_SHAPE_FILTER = True
EXCLUDED_VP_SHAPES = [""]

DISPLAY_MODE = "MONTHLY"
RESET_VP_PER_SESSION = True

USE_COOLDOWN_AFTER_LOSS = False
COOLDOWN_AFTER_LOSS_MINUTES = 60

EXCLUDED_HOURS_UTC = [0, 10]          # Heures UTC a exclure du trading
EXCLUDE_VAH_TARGET = True             # Exclure VAH (PD/PW) comme cible TP (statistiquement perdant)

SHOW_LAST_TRADES = False
LAST_TRADES_COUNT = 10
SHOW_OPEN_TRADES = False

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
        """Retourne un snapshot des niveaux actuels (pour historisation)."""
        poc, vah, val = self.get_levels()
        return {'poc': poc, 'vah': vah, 'val': val}


# =============================================================================
# STRUCTURAL LEVELS TRACKER
# Calcule et maintient les niveaux prev day & prev week VP
# =============================================================================
class StructuralLevelsTracker:
    """
    Construit les VP daily et weekly à partir des ticks, et expose
    les niveaux de la veille / semaine précédente.
    """
    def __init__(self, tick_size=0.01, va_percent=0.70):
        self.tick_size = tick_size
        self.va_percent = va_percent

        # VP accumulateurs pour le jour/semaine en cours
        self._daily_vp = IncrementalVolumeProfile(tick_size, va_percent)
        self._weekly_vp = IncrementalVolumeProfile(tick_size, va_percent)

        # Niveaux finalisés
        self.prev_day = {'poc': None, 'vah': None, 'val': None}
        self.prev_week = {'poc': None, 'vah': None, 'val': None}

        self._current_day = None
        self._current_week_start = None

    def update(self, dt, prices, volumes):
        """Appelé à chaque minute avec les ticks de cette minute."""
        day = dt.date()
        week_start = (dt - timedelta(days=dt.weekday())).date()

        # ── Changement de jour ──
        if self._current_day is not None and day != self._current_day:
            snap = self._daily_vp.snapshot()
            if snap['poc'] is not None:
                self.prev_day = snap
            self._daily_vp.reset()

        # ── Changement de semaine ──
        if self._current_week_start is not None and week_start != self._current_week_start:
            snap = self._weekly_vp.snapshot()
            if snap['poc'] is not None:
                self.prev_week = snap
            self._weekly_vp.reset()

        self._current_day = day
        self._current_week_start = week_start

        # Alimenter les VP courants
        if len(prices) > 0:
            self._daily_vp.add_ticks(prices, volumes)
            self._weekly_vp.add_ticks(prices, volumes)

    def get_target_levels(self, direction: str, entry_price: float):
        """
        Retourne la liste des niveaux structurels triés par distance depuis entry.
        direction: 'LONG' ou 'SHORT'
        Retourne: [(label, price), ...] triés du plus proche au plus loin.
        """
        candidates = []

        if USE_PREV_DAY_LEVELS:
            if self.prev_day['vah'] is not None and not EXCLUDE_VAH_TARGET:
                candidates.append(('PD_VAH', self.prev_day['vah']))
            if self.prev_day['val'] is not None:
                candidates.append(('PD_VAL', self.prev_day['val']))
            if self.prev_day['poc'] is not None:
                candidates.append(('PD_POC', self.prev_day['poc']))

        if USE_PREV_WEEK_LEVELS:
            if self.prev_week['vah'] is not None and not EXCLUDE_VAH_TARGET:
                candidates.append(('PW_VAH', self.prev_week['vah']))
            if self.prev_week['val'] is not None:
                candidates.append(('PW_VAL', self.prev_week['val']))
            if self.prev_week['poc'] is not None:
                candidates.append(('PW_POC', self.prev_week['poc']))

        # Filtrer : garder uniquement les niveaux dans la bonne direction
        if direction == 'LONG':
            valid = [(label, price) for label, price in candidates if price > entry_price]
            valid.sort(key=lambda x: x[1])  # Plus proche d'abord
        else:  # SHORT
            valid = [(label, price) for label, price in candidates if price < entry_price]
            valid.sort(key=lambda x: -x[1])  # Plus proche d'abord (le plus haut des prix < entry)

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
    # Charger 2 semaines avant pour avoir les prev week levels dès le début
    data_start = requested_start - timedelta(days=14)
    ts_start = int(data_start.timestamp() * 1000)

    sessions = asset.get('sessions', {'TOKYO': True, 'LONDON': True, 'NY': True})

    query_candles = f"SELECT ts, open, high, low, close FROM {asset['candle_table']} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df_candles = pd.read_sql(query_candles, conn)
    df_candles['dt'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)

    if df_candles.empty:
        return df_candles, pd.DataFrame()

    # On ne filtre PAS les candles par session ici :
    # on a besoin de toutes les candles pour construire les VP daily/weekly
    # Le filtrage session se fera dans la boucle principale

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
            return ("TP1_TP2", "TP1 ✓ → TP2 ✓ (Full Win)", "✅✅")
        elif tp2_hit:
            return ("TP2_DIRECT", "TP2 direct", "✅")
        else:
            return ("WIN_OTHER", "Win (autre)", "✅")
    elif result == "BE":
        if tp1_hit:
            return ("TP1_BE", "TP1 ✓ → BE (SL@Entry)", "🟡")
        else:
            return ("BE_OTHER", "BE (autre)", "🟡")
    elif result == "LOSS":
        if tp1_hit:
            return ("TP1_SL", "TP1 ✓ → SL (anormal)", "⚠️")
        else:
            return ("SL_DIRECT", "SL direct (Full Loss)", "❌")
    return ("UNKNOWN", "Unknown", "❓")


def display_last_trades(df_trades, count):
    if df_trades.empty:
        return
    last_trades = df_trades.tail(count)
    print(f"\n{'=' * 120}")
    print(f"DÉTAIL DES {len(last_trades)} DERNIERS TRADES")
    print(f"{'=' * 120}")
    tp1_pct = int(TP1_SPLIT * 100)
    tp2_pct = int(TP2_SPLIT * 100)

    for idx, trade in last_trades.iterrows():
        scenario_code, scenario_label, emoji = get_exit_scenario(trade)
        print(f"\n{'─' * 100}")
        print(f"TRADE #{idx + 1} | {trade['symbol']} | {trade['type']} | {emoji} {trade['result']} | {trade['pnl_r']:+.2f}R")
        print(f"{'─' * 100}")
        entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(trade['entry_time']) else "N/A"
        exit_time = trade['exit_time'].strftime('%H:%M') if pd.notna(trade['exit_time']) else "N/A"
        breakout_time = trade['breakout_time'].strftime('%H:%M') if pd.notna(trade['breakout_time']) else "N/A"
        print(f"  Session: {trade['session']} | Breakout: {breakout_time} | Entry: {entry_time} | Exit: {exit_time}")
        print(f"  Attente: {trade.get('confirmation_count', 'N/A')} bougies après breakout")
        print(f"\n  📊 SCÉNARIO: {scenario_label}")
        print(f"\n  PRIX:")
        print(f"    Entry:  {trade['entry']:.2f}")
        print(f"    SL:     {trade['sl']:.2f}")
        if pd.notna(trade.get('tp1')):
            tp1_status = "✓ HIT" if pd.notna(trade.get('tp1_time')) else "✗"
            print(f"    TP1:    {trade['tp1']:.2f} ({TP1_RR}R) {tp1_status}")
        tp2_status = "✓ HIT" if pd.notna(trade.get('tp2_time')) else "✗"
        print(f"    TP2:    {trade['tp']:.2f} ({trade.get('tp_label', '?')}) {tp2_status}")
        print(f"\n  VP SESSION:")
        print(f"    VAH: {trade['vah_at_entry']:.2f} | POC: {trade['poc_at_entry']:.2f} | VAL: {trade['val_at_entry']:.2f}")
        print(f"    POC Strength: {trade['poc_strength']:.2f}x | Shape: {trade['vp_shape']}")
        print(f"\n  STRUCTURAL LEVELS:")
        print(f"    TP target: {trade.get('tp_label', 'N/A')} = {trade['tp']:.2f}")
        prev_day_str = f"VAH={trade.get('pd_vah', 0):.2f} POC={trade.get('pd_poc', 0):.2f} VAL={trade.get('pd_val', 0):.2f}" if trade.get('pd_vah') else "N/A"
        prev_week_str = f"VAH={trade.get('pw_vah', 0):.2f} POC={trade.get('pw_poc', 0):.2f} VAL={trade.get('pw_val', 0):.2f}" if trade.get('pw_vah') else "N/A"
        print(f"    Prev Day:  {prev_day_str}")
        print(f"    Prev Week: {prev_week_str}")
        print(f"\n  RÉSULTAT:")
        print(f"    R:R: {trade['rr']:.2f} | PnL: {trade['pnl_r']:+.2f}R (${trade['pnl']:+.2f})")
        print(f"    Capital après: ${trade['capital_after']:,.2f}")


# =============================================================================
# MAIN BACKTEST
# =============================================================================
def run_backtest():
    conn = get_db_connection()
    enabled_assets = [a for a in ASSETS if a.get('enabled', True)]
    if not enabled_assets:
        print("[ERROR] Aucun asset activé.")
        return

    print("[DATA] Chargement des données...")
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
                'breakout_direction': None,  # 'UP' ou 'DOWN'
                'breakout_time': None,
                'breakout_level': None,      # VAH ou VAL franchi
                'candles_since_breakout': 0, # Compteur de bougies depuis le breakout
                'wait_highs': [],            # Highs pendant l'attente (pour swing SL)
                'wait_lows': [],             # Lows pendant l'attente (pour swing SL)
                # Cooldown
                'last_loss_time': None,
                'last_loss_direction': None,
            }
            print(f"   + {asset['symbol']}: {len(df_candles):,} candles | {len(df_ticks):,} ticks")
    conn.close()
    print(f"   [TIME] Chargé en {time.time() - t0:.2f}s")

    if not assets_data:
        print("[ERROR] Aucune donnée chargée.")
        return

    # Construire le DataFrame unifié de toutes les candles
    all_candles = []
    for symbol, data in assets_data.items():
        df = data['candles'].copy()
        df['symbol'] = symbol
        all_candles.append(df)
    df_all_candles = pd.concat(all_candles).sort_values(['dt', 'symbol']).reset_index(drop=True)

    current_capital = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    max_dd_amount = 0.0
    max_dd_percent = 0.0
    all_trades = []

    # Compteurs de filtrage
    total_potential_entries = 0
    filtered_by_no_target = 0
    filtered_by_rr = 0
    filtered_by_poc_strength = 0
    filtered_by_vp_shape = 0
    filtered_by_cooldown = 0

    # Tracking périodique
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
    tp1_pct = int(TP1_SPLIT * 100)
    tp2_pct = int(TP2_SPLIT * 100)

    # ── HEADER ──
    print(f"\n[BACKTEST] VP Confirmed Breakout (Momentum) | ASSETS: {symbols_list}")
    print(f"[CAPITAL] ${INITIAL_CAPITAL:,.2f} | Risque: {RISK_PERCENT*100}%")
    print(f"[ENTRY]   Attente: {WAIT_CANDLES} bougies apres breakout, si toujours dehors -> entree")
    print(f"[SL]      Swing low/high des bougies de confirmation + {SL_OFFSET_POINTS} pts")
    print(f"[TP]      Niveau structurel le plus proche (R:R ≥ {MIN_RR})")
    print(f"[SPLIT]   TP1/TP2: {tp1_pct}/{tp2_pct} | TP1 @ {TP1_RR}R | Trailing: {'ON' if USE_TRAILING else 'OFF'}")
    print(f"[LEVELS]  Prev Day: {'ON' if USE_PREV_DAY_LEVELS else 'OFF'} | Prev Week: {'ON' if USE_PREV_WEEK_LEVELS else 'OFF'}")
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

        # ── Période tracking ──
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
        # ÉTAPE 1: STRUCTURAL LEVELS — alimenter les VP daily/weekly
        # (Toujours, avec TOUTES les candles, pas de filtrage session)
        # =================================================================
        if current_minute in ad['ticks_by_minute']:
            prices, volumes, timestamps = ad['ticks_by_minute'][current_minute]
            ad['structural'].update(row.dt, prices, volumes)

        # Skip si avant la date de début (période de warmup pour les niveaux)
        if row.dt < requested_start:
            continue

        # =================================================================
        # ÉTAPE 2: SESSION VP (uniquement pendant les sessions actives)
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

        # Ajouter les ticks au VP session
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
        # ÉTAPE 3: GESTION DES TRADES ACTIFS
        # =================================================================
        active_trade = ad['active_trade']
        if active_trade:
            res = None
            if active_trade['type'] == 'LONG':
                if row.low <= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    if USE_TRAILING and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] + (active_trade['risk'] * TP1_RR)
                        if row.high >= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = TP1_RR * TP1_SPLIT
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
                    if USE_TRAILING and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] - (active_trade['risk'] * TP1_RR)
                        if row.low <= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = TP1_RR * TP1_SPLIT
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
                    if USE_TRAILING and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r + (trade_rr * TP2_SPLIT)
                    else:
                        pnl_r = trade_rr
                    pnl = risk_amount * pnl_r
                    day_wins += 1
                else:
                    if USE_TRAILING and active_trade.get('partial_closed', False):
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
                    'breakout_time': active_trade.get('breakout_time'),
                    'confirmation_count': active_trade.get('wait_candles'),
                    'confirmation_window': active_trade.get('wait_candles'),
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
        # ÉTAPE 4: STATE MACHINE — Breakout confirmé par le temps
        # INSIDE → WAITING (breakout détecté, on attend N bougies)
        #   → si après N bougies toujours dehors → ENTRY
        #   → si réintègre la VA avant N bougies → retour INSIDE
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
            # Détecter un nouveau breakout
            if close > vah:
                ad['state'] = "WAITING"
                ad['breakout_direction'] = "UP"
                ad['breakout_time'] = row.dt
                ad['breakout_level'] = vah
                ad['candles_since_breakout'] = 1  # Cette bougie est la 1ère
                ad['wait_highs'] = [high]
                ad['wait_lows'] = [low]
            elif close < val:
                ad['state'] = "WAITING"
                ad['breakout_direction'] = "DOWN"
                ad['breakout_time'] = row.dt
                ad['breakout_level'] = val
                ad['candles_since_breakout'] = 1
                ad['wait_highs'] = [high]
                ad['wait_lows'] = [low]

        elif state == "WAITING":
            ad['candles_since_breakout'] += 1
            ad['wait_highs'].append(high)
            ad['wait_lows'].append(low)

            # ── Breakout invalidé : le close réintègre la VA ──
            if ad['breakout_direction'] == "UP" and close <= vah:
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0
                continue
            if ad['breakout_direction'] == "DOWN" and close >= val:
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0
                continue

            # ── N bougies atteintes ET toujours dehors → ENTRÉE ──
            if ad['candles_since_breakout'] >= WAIT_CANDLES:
                total_potential_entries += 1

                direction = 'LONG' if ad['breakout_direction'] == 'UP' else 'SHORT'

                # Filtres
                poc_ok = True
                if USE_VP_STRUCTURE_FILTER:
                    if poc_strength is None or poc_strength < MIN_POC_STRENGTH:
                        poc_ok = False
                        filtered_by_poc_strength += 1

                shape_ok = True
                if USE_VP_SHAPE_FILTER:
                    if vp_shape in EXCLUDED_VP_SHAPES:
                        shape_ok = False
                        filtered_by_vp_shape += 1

                cooldown_ok = True
                if USE_COOLDOWN_AFTER_LOSS and ad['last_loss_time'] is not None:
                    if ad['last_loss_direction'] == direction:
                        mins = (row.dt - ad['last_loss_time']).total_seconds() / 60.0
                        if mins < COOLDOWN_AFTER_LOSS_MINUTES:
                            cooldown_ok = False
                            filtered_by_cooldown += 1

                can_direction = (direction == 'LONG' and config['allow_long']) or (direction == 'SHORT' and config['allow_short'])
                can_time = can_trade_now(row.dt, curr_sess)
                day_ok = row.dt.weekday() in config.get('allowed_days', [0,1,2,3,4])
                hour_ok = row.dt.hour not in EXCLUDED_HOURS_UTC

                if poc_ok and shape_ok and cooldown_ok and can_direction and can_time and day_ok and hour_ok:
                    # ── SL : swing low/high des bougies d'attente ──
                    if direction == 'LONG':
                        swing_low = min(ad['wait_lows'])
                        sl = swing_low - SL_OFFSET_POINTS
                        risk = close - sl
                    else:
                        swing_high = max(ad['wait_highs'])
                        sl = swing_high + SL_OFFSET_POINTS
                        risk = sl - close

                    if risk > 0:
                        # ── TP : niveau structurel le plus proche avec R:R >= MIN_RR ──
                        targets = ad['structural'].get_target_levels(direction, close)

                        tp = None
                        tp_label = None
                        for label, price in targets:
                            if direction == 'LONG':
                                rr = (price - close) / risk
                            else:
                                rr = (close - price) / risk
                            if rr >= MIN_RR and rr <= 30:
                                tp = price
                                tp_label = label
                                break

                        # Fallback : pas de niveau structurel → TP fixe à MIN_RR
                        if tp is None:
                            filtered_by_no_target += 1
                            if direction == 'LONG':
                                tp = close + (risk * MIN_RR)
                            else:
                                tp = close - (risk * MIN_RR)
                            tp_label = f"FIXED_{MIN_RR}R"
                            actual_rr = MIN_RR
                        else:
                            if direction == 'LONG':
                                actual_rr = (tp - close) / risk
                            else:
                                actual_rr = (close - tp) / risk

                        tp1_price = None
                        if USE_TRAILING:
                            if direction == 'LONG':
                                tp1_price = close + (risk * TP1_RR)
                            else:
                                tp1_price = close - (risk * TP1_RR)

                        pd_levels = ad['structural'].prev_day
                        pw_levels = ad['structural'].prev_week

                        ad['active_trade'] = {
                            'type': direction, 'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
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

                # Reset après tentative
                ad['state'] = "INSIDE"
                ad['breakout_direction'] = None
                ad['candles_since_breakout'] = 0

    # ── Dernière période ──
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
    # RAPPORT FINAL
    # =================================================================
    print("\n" + "=" * 120)
    print("[RAPPORT FINAL] VP Confirmed Breakout")
    print("=" * 120)

    if not all_trades:
        print("[ERROR] Aucun trade exécuté.")
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

    # Scénarios de sortie
    print(f"\n{'─' * 60}")
    print("SCÉNARIOS DE SORTIE")
    print(f"{'─' * 60}")
    tp1_tp2 = len(df_trades[(df_trades['result']=='WIN') & (df_trades['tp1_time'].notna())])
    tp2_direct = len(df_trades[(df_trades['result']=='WIN') & (df_trades['tp1_time'].isna())])
    sl_direct = len(df_trades[(df_trades['result']=='LOSS') & (df_trades['tp1_time'].isna())])
    print(f"  TP1 → TP2 (full win):  {tp1_tp2:>5}")
    print(f"  TP1 → BE:              {be_trades:>5}")
    print(f"  SL direct:             {sl_direct:>5}")
    if tp2_direct > 0:
        print(f"  TP2 direct:            {tp2_direct:>5}")

    # Filtres
    print(f"\n{'─' * 60}")
    print("FILTRES APPLIQUÉS")
    print(f"{'─' * 60}")
    print(f"  Breakouts confirmés:  {total_potential_entries:>14}")
    print(f"  Fallback fixe (no lvl):{filtered_by_no_target:>13} ({filtered_by_no_target/max(1,total_potential_entries)*100:.1f}%)")
    print(f"  POC faible:           {filtered_by_poc_strength:>14}")
    print(f"  VP Shape:             {filtered_by_vp_shape:>14}")

    # Performance par TP target
    if 'tp_label' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR NIVEAU CIBLE")
        print(f"{'─' * 60}")
        for label in ['PD_VAH', 'PD_POC', 'PD_VAL', 'PW_VAH', 'PW_POC', 'PW_VAL']:
            lt = df_trades[df_trades['tp_label'] == label]
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

    # Breakdown jour
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

    # Breakdown heure
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
    print(f"  Max Série Gagnante:   {max_winning_streak:>14}")
    print(f"  Max Série Perdante:   {max_losing_streak:>14}")
    print(f"  Temps d'exécution:    {elapsed:>13.2f}s")

    if SHOW_LAST_TRADES:
        display_last_trades(df_trades, LAST_TRADES_COUNT)

    print("=" * 120)


if __name__ == "__main__":
    run_backtest()
