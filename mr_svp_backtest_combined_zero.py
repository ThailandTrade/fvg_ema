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
# CONFIGURATION MULTI-ASSETS
# =============================================================================

START_DATE_STR = "2025-05-01 00:00:00"
INITIAL_CAPITAL = 100000.0
RISK_PERCENT = 0.008

TP_MODE = "POC"
TARGET_RR = 3.0
MIN_RR = 2.0

USE_TRAILING = True
TP1_RR = 1.3

FILTER_ENTRY_VS_POC = True
USE_BREAKOUT_DURATION_FILTER = True
MAX_BREAKOUT_DURATION_MINUTES = 3
USE_VP_STRUCTURE_FILTER = True
MIN_POC_STRENGTH = 3.0
USE_VP_SHAPE_FILTER = True
EXCLUDED_VP_SHAPES = [""]
DISPLAY_MODE = "WEEKLY"
RESET_VP_PER_SESSION = True

# =============================================================================
# RÈGLES PROP FIRM
# =============================================================================
PROP_FIRM_RULES = {
    'max_daily_dd_percent': 3.0,      # Max Daily DD < 3%
    'max_total_dd_percent': 5.0,      # Max DD < 5%
    'max_day_profit_percent': 15.0,   # Un jour ne peut pas dépasser 15% du profit total
    'min_positive_days': 7,           # Au moins 7 jours avec gain >= min_daily_gain_percent
    'min_daily_gain_percent': 0.25,   # Gain minimum par jour pour compter comme "jour qualifiant"
    'rolling_window_days': 30,        # Sur une période de 30 jours
    'min_trade_duration_above_1min_percent': 50.0,  # 50% des trades doivent durer > 1 minute
}

# =============================================================================
# AFFICHAGE DES DERNIERS TRADES
# =============================================================================
SHOW_LAST_TRADES = False
LAST_TRADES_COUNT = 10

# =============================================================================
# AFFICHAGE DES TRADES EN COURS (NON CLOS)
# =============================================================================
SHOW_OPEN_TRADES = False

# =============================================================================
# CONFIGURATION DES HEURES DE SESSION (UTC)
# =============================================================================
SESSIONS_CONFIG = {
    'TOKYO':  {'vp_start': 0,    'vp_end': 4,    'trade_start': 0,    'trade_end': 4},
    'LONDON': {'vp_start': 8,    'vp_end': 13,   'trade_start': 9,    'trade_end': 13},   # VP dès 8h, trades dès 9h
    'NY':     {'vp_start': 14.5, 'vp_end': 21,   'trade_start': 14.5, 'trade_end': 21},
}

# SESSIONS_CONFIG = {
    # 'TOKYO':  {'vp_start': 0,    'vp_end': 4,    'trade_start': 0,    'trade_end': 4},
    # 'LONDON': {'vp_start': 8,    'vp_end': 13,   'trade_start': 9,    'trade_end': 13},
    # 'NY':     {'vp_start': 14.5, 'vp_end': 21,   'trade_start': 14.5, 'trade_end': 21},
# }

ASSETS = [
    {
        'enabled': True,
        'symbol': 'XAUUSD',
        'candle_table': 'candles_mt5_xauusd_1m',
        'tick_table': 'market_ticks_xauusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'sl_offset': 0.50,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
    },
    {
        'enabled': False,
        'symbol': 'US100.cash',
        'candle_table': 'candles_mt5_us100_cash_1m',
        'tick_table': 'market_ticks_us100',
        'tick_size': 1.0,
        'va_percent': 0.70,
        'sl_offset': 1.0,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
    },
    {
        'enabled': False,
        'symbol': 'GER40.cash',
        'candle_table': 'candles_mt5_ger40_cash_1m',
        'tick_table': 'market_ticks_ger40',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'sl_offset': 1.0,
        'allow_long': True,
        'allow_short': False,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
    },
    {
        'enabled': False,
        'symbol': 'XAUAUD',
        'candle_table': 'candles_mt5_xauaud_1m',
        'tick_table': 'market_ticks_xauaud',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'sl_offset': 0.50,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': True},
    },
]

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


def get_session_start_time(session_name: str, reference_dt: datetime) -> datetime:
    if session_name not in SESSIONS_CONFIG:
        return None
    cfg = SESSIONS_CONFIG[session_name]
    start_hour = cfg['vp_start']
    hour = int(start_hour)
    minute = int((start_hour % 1) * 60)
    return reference_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def is_in_session_vp(dt: datetime, session_name: str) -> bool:
    if session_name not in SESSIONS_CONFIG:
        return False
    cfg = SESSIONS_CONFIG[session_name]
    current_time = dt.hour + dt.minute / 60.0
    return cfg['vp_start'] <= current_time < cfg['vp_end']


def is_in_session_trade(dt: datetime, session_name: str) -> bool:
    if session_name not in SESSIONS_CONFIG:
        return False
    cfg = SESSIONS_CONFIG[session_name]
    current_time = dt.hour + dt.minute / 60.0
    return cfg['trade_start'] <= current_time < cfg['trade_end']


def load_all_data(conn, asset):
    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = requested_start - timedelta(hours=2)
    ts_start = int(data_start.timestamp() * 1000)
    
    sessions = asset.get('sessions', {'TOKYO': True, 'LONDON': True, 'NY': True})
    
    query_candles = f"SELECT ts, open, high, low, close FROM {asset['candle_table']} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df_candles = pd.read_sql(query_candles, conn)
    df_candles['dt'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)
    
    if df_candles.empty:
        return df_candles, pd.DataFrame()
    
    def is_candle_in_active_session(dt):
        current_time = dt.hour + dt.minute / 60.0
        for sess_name, is_active in sessions.items():
            if is_active and sess_name in SESSIONS_CONFIG:
                cfg = SESSIONS_CONFIG[sess_name]
                if cfg['vp_start'] <= current_time < cfg['vp_end']:
                    return True
        return False
    
    df_candles = df_candles[df_candles['dt'].apply(is_candle_in_active_session)].reset_index(drop=True)
    
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


def can_trade_now(dt, session_name: str) -> bool:
    if session_name not in SESSIONS_CONFIG:
        return False
    cfg = SESSIONS_CONFIG[session_name]
    current_time = dt.hour + dt.minute / 60.0
    return cfg['trade_start'] <= current_time < cfg['trade_end']


def get_exit_scenario(trade):
    result = trade['result']
    tp1_hit = pd.notna(trade.get('tp1_time'))
    tp2_hit = pd.notna(trade.get('tp2_time'))
    
    if result == "WIN":
        if tp1_hit and tp2_hit:
            return ("TP1_TP2", "TP1 OK -> TP2 OK (Full Win)", "[WIN]")
        elif tp2_hit:
            return ("TP2_DIRECT", "TP2 direct (rare)", "[WIN]")
        else:
            return ("WIN_OTHER", "Win (autre)", "[WIN]")
    
    elif result == "BE":
        if tp1_hit:
            return ("TP1_BE", "TP1 OK -> BE (SL@Entry)", "[BE]")
        else:
            return ("BE_OTHER", "BE (autre)", "[BE]")
    
    elif result == "LOSS":
        if tp1_hit:
            return ("TP1_SL", "TP1 OK -> SL (anormal)", "[WARN]")
        else:
            return ("SL_DIRECT", "SL direct (Full Loss)", "[LOSS]")
    
    return ("UNKNOWN", "Unknown", "[?]")


def display_last_trades(df_trades, count):
    if df_trades.empty:
        return
    
    last_trades = df_trades.tail(count)
    
    print(f"\n{'=' * 120}")
    print(f"DÉTAIL DES {len(last_trades)} DERNIERS TRADES")
    print(f"{'=' * 120}")
    
    for idx, trade in last_trades.iterrows():
        scenario_code, scenario_label, emoji = get_exit_scenario(trade)
        
        print(f"\n{'─' * 100}")
        print(f"TRADE #{idx + 1} | {trade['symbol']} | {trade['type']} | {emoji} {trade['result']} | {trade['pnl_r']:+.2f}R | Duration: {trade['trade_duration_min']:.1f}min")
        print(f"{'─' * 100}")
        
        entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(trade['entry_time']) else "N/A"
        exit_time = trade['exit_time'].strftime('%H:%M') if pd.notna(trade['exit_time']) else "N/A"
        breakout_time = trade['breakout_time'].strftime('%H:%M') if pd.notna(trade['breakout_time']) else "N/A"
        
        print(f"  Session: {trade['session']} | Breakout: {breakout_time} | Entry: {entry_time} | Exit: {exit_time}")
        print(f"  Breakout Duration: {trade['breakout_duration_min']:.1f} min | Trade Duration: {trade['trade_duration_min']:.1f} min")
        
        print(f"\n  SCÉNARIO DE SORTIE: {scenario_label}")
        
        tp1_time = trade['tp1_time'].strftime('%H:%M') if pd.notna(trade.get('tp1_time')) else None
        tp2_time = trade['tp2_time'].strftime('%H:%M') if pd.notna(trade.get('tp2_time')) else None
        
        timeline = f"     Entry ({entry_time[-5:]}) "
        
        if scenario_code == "TP1_TP2":
            timeline += f"→ TP1 ({tp1_time}) → TP2 ({tp2_time})"
        elif scenario_code == "TP1_BE":
            timeline += f"→ TP1 ({tp1_time}) → BE ({exit_time})"
        elif scenario_code == "SL_DIRECT":
            timeline += f"→ SL ({exit_time})"
        elif scenario_code == "TP2_DIRECT":
            timeline += f"→ TP2 ({tp2_time})"
        else:
            timeline += f"→ Exit ({exit_time})"
        
        print(timeline)
        
        print(f"\n  PRIX:")
        print(f"    Entry:  {trade['entry']:.2f}")
        
        sl_status = "← HIT" if scenario_code == "SL_DIRECT" else ("← moved to BE" if scenario_code == "TP1_BE" else "")
        print(f"    SL:     {trade['sl']:.2f} {sl_status}")
        
        if pd.notna(trade.get('tp1')):
            tp1_status = "[OK] HIT" if pd.notna(trade.get('tp1_time')) else "[X] not hit"
            print(f"    TP1:    {trade['tp1']:.2f} ({TP1_RR}R) {tp1_status}")
        
        tp2_status = "[OK] HIT" if pd.notna(trade.get('tp2_time')) else "[X] not hit"
        print(f"    TP2:    {trade['tp']:.2f} (POC) {tp2_status}")
        
        exit_label = ""
        if trade['exit_price'] == trade['sl']:
            exit_label = "(= SL)"
        elif trade['exit_price'] == trade['entry']:
            exit_label = "(= Entry/BE)"
        elif trade['exit_price'] == trade['tp']:
            exit_label = "(= TP2)"
        print(f"    Exit:   {trade['exit_price']:.2f} {exit_label}")
        
        print(f"\n  VOLUME PROFILE:")
        print(f"    VAH: {trade['vah_at_entry']:.2f} | POC: {trade['poc_at_entry']:.2f} | VAL: {trade['val_at_entry']:.2f}")
        print(f"    POC Strength: {trade['poc_strength']:.2f}x | Shape: {trade['vp_shape']}")
        
        risk_pts = abs(trade['entry'] - trade['sl'])
        print(f"\n  RÉSULTAT:")
        print(f"    R:R Potentiel: {trade['rr']:.2f}")
        print(f"    Risk (pts): {risk_pts:.2f}")
        
        if scenario_code == "TP1_TP2":
            pnl_tp1 = TP1_RR * 0.5
            pnl_tp2 = trade['rr'] * 0.5
            print(f"    PnL TP1 (50%): +{pnl_tp1:.2f}R")
            print(f"    PnL TP2 (50%): +{pnl_tp2:.2f}R")
            print(f"    PnL Total: {trade['pnl_r']:+.2f}R (${trade['pnl']:+.2f})")
        elif scenario_code == "TP1_BE":
            pnl_tp1 = TP1_RR * 0.5
            print(f"    PnL TP1 (50%): +{pnl_tp1:.2f}R")
            print(f"    PnL TP2 (50%): 0.00R (BE)")
            print(f"    PnL Total: {trade['pnl_r']:+.2f}R (${trade['pnl']:+.2f})")
        elif scenario_code == "SL_DIRECT":
            print(f"    PnL TP1 (50%): -0.50R")
            print(f"    PnL TP2 (50%): -0.50R")
            print(f"    PnL Total: {trade['pnl_r']:+.2f}R (${trade['pnl']:+.2f})")
        else:
            print(f"    PnL: {trade['pnl_r']:+.2f}R (${trade['pnl']:+.2f})")
        
        print(f"    Capital après: ${trade['capital_after']:,.2f}")


def display_open_trades(assets_data, current_capital):
    open_trades = []
    
    for symbol, data in assets_data.items():
        if data['active_trade'] is not None:
            trade = data['active_trade'].copy()
            trade['symbol'] = symbol
            open_trades.append(trade)
    
    if not open_trades:
        print(f"\n{'=' * 80}")
        print("TRADES EN COURS: Aucun")
        print(f"{'=' * 80}")
        return
    
    print(f"\n{'=' * 120}")
    print(f"TRADES EN COURS ({len(open_trades)} positions ouvertes)")
    print(f"{'=' * 120}")
    
    for trade in open_trades:
        symbol = trade['symbol']
        trade_type = trade['type']
        entry = trade['entry']
        sl = trade['sl']
        original_sl = trade.get('original_sl', sl)
        tp1 = trade.get('tp1')
        tp2 = trade['tp']
        rr = trade['rr']
        partial_closed = trade.get('partial_closed', False)
        
        entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M') if trade.get('entry_time') else "N/A"
        breakout_time = trade['breakout_time'].strftime('%H:%M') if trade.get('breakout_time') else "N/A"
        
        if partial_closed:
            status = "[PARTIAL] TP1 HIT - SL @ BE"
            sl_display = f"{sl:.2f} (moved to BE)"
        else:
            status = "[OPEN] En attente TP1"
            sl_display = f"{sl:.2f}"
        
        print(f"\n{'─' * 100}")
        print(f"OPEN | {symbol} | {trade_type} | {status}")
        print(f"{'─' * 100}")
        
        print(f"  Session: {trade.get('session_at_open', 'N/A')} | Breakout: {breakout_time} | Entry: {entry_time}")
        
        print(f"\n  PRIX:")
        print(f"    Entry:      {entry:.2f}")
        print(f"    SL:         {sl_display}")
        if tp1:
            tp1_status = "[OK] HIT" if partial_closed else "[PENDING]"
            print(f"    TP1:        {tp1:.2f} ({TP1_RR}R) {tp1_status}")
        print(f"    TP2 (POC):  {tp2:.2f} [PENDING]")
        
        print(f"\n  VOLUME PROFILE:")
        print(f"    VAH: {trade.get('vah_at_entry', 0):.2f} | POC: {trade.get('poc_at_entry', 0):.2f} | VAL: {trade.get('val_at_entry', 0):.2f}")
        print(f"    POC Strength: {trade.get('poc_strength', 0):.2f}x | Shape: {trade.get('vp_shape', 'N/A')}")
        
        print(f"\n  POTENTIEL:")
        print(f"    R:R Potentiel: {rr:.2f}")
        risk_amount = current_capital * RISK_PERCENT
        
        if partial_closed:
            locked_pnl = TP1_RR * 0.5
            remaining_potential = rr * 0.5
            print(f"    PnL verrouillé (TP1): +{locked_pnl:.2f}R")
            print(f"    Potentiel restant (TP2): +{remaining_potential:.2f}R")
            print(f"    Risque restant: 0R (SL @ BE)")
        else:
            print(f"    Si TP1+TP2: +{(TP1_RR * 0.5 + rr * 0.5):.2f}R (${risk_amount * (TP1_RR * 0.5 + rr * 0.5):+.2f})")
            print(f"    Si TP1+BE:  +{TP1_RR * 0.5:.2f}R (${risk_amount * TP1_RR * 0.5:+.2f})")
            print(f"    Si SL:      -1.00R (${-risk_amount:.2f})")


def check_prop_firm_rules(df_trades, initial_capital):
    """
    Vérifie la conformité aux règles prop firm
    Retourne un dictionnaire avec les résultats de chaque vérification
    """
    rules = PROP_FIRM_RULES
    results = {
        'compliant': True,
        'violations': [],
        'checks': {}
    }
    
    if df_trades.empty:
        return results
    
    # Créer une copie pour ne pas modifier le DataFrame original
    df_trades_copy = df_trades.copy()
    
    # Préparer les données par jour
    df_trades_copy['date_only'] = pd.to_datetime(df_trades_copy['date']).dt.date
    daily_stats = df_trades_copy.groupby('date_only').agg({
        'pnl': 'sum',
        'capital_after': 'last'
    }).reset_index()
    
    # Calculer le capital de début de journée pour chaque jour
    daily_stats['day_start_capital'] = daily_stats['capital_after'].shift(1).fillna(initial_capital)
    
    # 1. Vérifier Max Daily DD < 3%
    daily_stats['day_high'] = daily_stats.groupby(daily_stats.index)['capital_after'].cummax()
    daily_stats['daily_dd_amount'] = 0.0
    daily_stats['daily_dd_percent'] = 0.0
    
    for date in daily_stats['date_only'].unique():
        day_trades = df_trades_copy[df_trades_copy['date_only'] == date].sort_values('entry_time')
        if len(day_trades) == 0:
            continue
        
        day_start = daily_stats[daily_stats['date_only'] == date]['day_start_capital'].iloc[0]
        day_high = day_start
        max_dd = 0.0
        
        for _, trade in day_trades.iterrows():
            if trade['capital_after'] > day_high:
                day_high = trade['capital_after']
            dd = day_high - trade['capital_after']
            if dd > max_dd:
                max_dd = dd
        
        dd_percent = (max_dd / day_high * 100) if day_high > 0 else 0
        daily_stats.loc[daily_stats['date_only'] == date, 'daily_dd_amount'] = max_dd
        daily_stats.loc[daily_stats['date_only'] == date, 'daily_dd_percent'] = dd_percent
        
        if dd_percent > rules['max_daily_dd_percent']:
            results['compliant'] = False
            results['violations'].append(
                f"[X] Daily DD violated on {date}: {dd_percent:.2f}% (max: {rules['max_daily_dd_percent']}%)"
            )
    
    max_daily_dd = daily_stats['daily_dd_percent'].max()
    results['checks']['max_daily_dd'] = {
        'value': max_daily_dd,
        'limit': rules['max_daily_dd_percent'],
        'passed': max_daily_dd <= rules['max_daily_dd_percent']
    }
    
    # 2. Vérifier Max Total DD < 5%
    max_total_dd = df_trades['drawdown'].max()
    max_total_dd_percent = (max_total_dd / df_trades['high_water_mark'].max() * 100) if df_trades['high_water_mark'].max() > 0 else 0
    
    results['checks']['max_total_dd'] = {
        'value': max_total_dd_percent,
        'limit': rules['max_total_dd_percent'],
        'passed': max_total_dd_percent <= rules['max_total_dd_percent']
    }
    
    if max_total_dd_percent > rules['max_total_dd_percent']:
        results['compliant'] = False
        results['violations'].append(
            f"[X] Total DD violated: {max_total_dd_percent:.2f}% (max: {rules['max_total_dd_percent']}%)"
        )
    
    # 3. Vérifier qu'aucun jour ne dépasse 15% du profit total
    total_profit = df_trades['pnl'].sum()
    if total_profit > 0:
        daily_stats['day_profit_percent'] = (daily_stats['pnl'] / total_profit * 100).abs()
        max_day_contribution = daily_stats['day_profit_percent'].max()
        
        results['checks']['max_day_profit'] = {
            'value': max_day_contribution,
            'limit': rules['max_day_profit_percent'],
            'passed': max_day_contribution <= rules['max_day_profit_percent']
        }
        
        if max_day_contribution > rules['max_day_profit_percent']:
            worst_day = daily_stats.loc[daily_stats['day_profit_percent'].idxmax(), 'date']
            results['compliant'] = False
            results['violations'].append(
                f"[X] Single day profit too high on {worst_day}: {max_day_contribution:.2f}% of total (max: {rules['max_day_profit_percent']}%)"
            )
    else:
        results['checks']['max_day_profit'] = {
            'value': 0,
            'limit': rules['max_day_profit_percent'],
            'passed': True
        }
    
    # 4. Vérifier au moins 7 jours avec gain >= 0.25% sur toute période de 30 jours
    min_gain_threshold = rules.get('min_daily_gain_percent', 0.25)
    daily_stats['day_gain_percent'] = (daily_stats['pnl'] / daily_stats['day_start_capital'] * 100)
    daily_stats['is_qualifying_day'] = daily_stats['day_gain_percent'] >= min_gain_threshold
    
    all_dates = pd.date_range(start=daily_stats['date_only'].min(), end=daily_stats['date_only'].max(), freq='D')
    daily_complete = pd.DataFrame({'date_only': all_dates.date})
    daily_complete = daily_complete.merge(daily_stats[['date_only', 'is_qualifying_day']], on='date_only', how='left')
    daily_complete['is_qualifying_day'] = daily_complete['is_qualifying_day'].fillna(False)
    
    min_qualifying_days_in_window = float('inf')
    worst_window_start = None
    total_days = len(daily_complete)
    
    if total_days >= 30:
        for i in range(len(daily_complete) - 29):
            window = daily_complete.iloc[i:i+30]
            qualifying_days = window['is_qualifying_day'].sum()
            
            if qualifying_days < min_qualifying_days_in_window:
                min_qualifying_days_in_window = qualifying_days
                worst_window_start = window['date_only'].iloc[0]
        
        results['checks']['min_positive_days'] = {
            'value': int(min_qualifying_days_in_window),
            'limit': rules['min_positive_days'],
            'passed': min_qualifying_days_in_window >= rules['min_positive_days']
        }
        
        if min_qualifying_days_in_window < rules['min_positive_days']:
            results['compliant'] = False
            results['violations'].append(
                f"[X] Insufficient qualifying days (>= +{min_gain_threshold}%) in 30-day window starting {worst_window_start}: {int(min_qualifying_days_in_window)} days (min: {rules['min_positive_days']})"
            )
    else:
        total_qualifying_days = daily_complete['is_qualifying_day'].sum()
        results['checks']['min_positive_days'] = {
            'value': int(total_qualifying_days),
            'limit': rules['min_positive_days'],
            'passed': None,
            'note': f'Insufficient data for 30-day window ({total_days} days total)'
        }
    
    # 5. Vérifier que >= 50% des trades durent > 1 minute
    min_duration_threshold = rules.get('min_trade_duration_above_1min_percent', 50.0)
    if 'trade_duration_min' in df_trades.columns:
        trades_above_1min = (df_trades['trade_duration_min'] > 1.0).sum()
        total_trades = len(df_trades)
        pct_above_1min = (trades_above_1min / total_trades * 100) if total_trades > 0 else 0
        
        results['checks']['trade_duration'] = {
            'value': pct_above_1min,
            'limit': min_duration_threshold,
            'passed': pct_above_1min >= min_duration_threshold,
            'trades_above_1min': int(trades_above_1min),
            'total_trades': total_trades
        }
        
        if pct_above_1min < min_duration_threshold:
            results['compliant'] = False
            results['violations'].append(
                f"[X] Trade duration rule violated: {pct_above_1min:.1f}% trades > 1min (min: {min_duration_threshold}%)"
            )
    
    return results


def display_prop_firm_compliance(compliance_results):
    """Affiche les résultats de conformité prop firm"""
    print(f"\n{'=' * 120}")
    print("VÉRIFICATION CONFORMITÉ PROP FIRM")
    print(f"{'=' * 120}")
    
    if compliance_results['compliant']:
        print("\n[OK] CONFORME - Toutes les règles prop firm sont respectées!\n")
    else:
        print("\n[FAIL] NON CONFORME - Violations détectées:\n")
        for violation in compliance_results['violations']:
            print(f"   {violation}")
        print()
    
    print(f"{'─' * 60}")
    print("DÉTAIL DES VÉRIFICATIONS")
    print(f"{'─' * 60}")
    
    # 1. Max Daily DD
    check = compliance_results['checks'].get('max_daily_dd', {})
    status = "[OK]" if check.get('passed') else "[FAIL]"
    print(f"\n{status} Max Daily Drawdown:")
    print(f"   Valeur: {check.get('value', 0):.2f}% | Limite: {check.get('limit', 0):.2f}%")
    
    # 2. Max Total DD
    check = compliance_results['checks'].get('max_total_dd', {})
    status = "[OK]" if check.get('passed') else "[FAIL]"
    print(f"\n{status} Max Total Drawdown:")
    print(f"   Valeur: {check.get('value', 0):.2f}% | Limite: {check.get('limit', 0):.2f}%")
    
    # 3. Max Day Profit
    check = compliance_results['checks'].get('max_day_profit', {})
    status = "[OK]" if check.get('passed') else "[FAIL]"
    print(f"\n{status} Max Single Day Profit:")
    print(f"   Valeur: {check.get('value', 0):.2f}% du total | Limite: {check.get('limit', 0):.2f}%")
    
    # 4. Min Positive Days
    check = compliance_results['checks'].get('min_positive_days', {})
    if check.get('passed') is None:
        status = "[WARN]"
        note = f" ({check.get('note', 'N/A')})"
    else:
        status = "[OK]" if check.get('passed') else "[FAIL]"
        note = ""
    min_gain = PROP_FIRM_RULES.get('min_daily_gain_percent', 0.25)
    print(f"\n{status} Jours Qualifiants (>= +{min_gain}% sur 30j):")
    print(f"   Valeur: {check.get('value', 0)} jours | Minimum: {check.get('limit', 0)} jours{note}")
    
    # 5. Trade Duration
    check = compliance_results['checks'].get('trade_duration', {})
    if check:
        status = "[OK]" if check.get('passed') else "[FAIL]"
        print(f"\n{status} Durée des Trades (> 1 minute):")
        print(f"   Trades > 1min: {check.get('trades_above_1min', 0)} / {check.get('total_trades', 0)} ({check.get('value', 0):.1f}%)")
        print(f"   Minimum requis: {check.get('limit', 0):.0f}%")
    
    print(f"\n{'=' * 120}")


def run_backtest():
    conn = get_db_connection()
    enabled_assets = [a for a in ASSETS if a.get('enabled', True)]
    if not enabled_assets:
        print("[ERROR] Aucun asset active.")
        return
    print("[DATA] Chargement des donnees...")
    t0 = time.time()
    assets_data = {}
    for asset in enabled_assets:
        df_candles, df_ticks = load_all_data(conn, asset)
        if not df_candles.empty:
            df_ticks['minute'] = df_ticks['time'].dt.floor('T')
            ticks_by_minute = df_ticks.groupby('minute').apply(
                lambda g: (g['price'].values, g['volume'].values, g['time'].values)
            ).to_dict()
            
            assets_data[asset['symbol']] = {
                'config': asset, 'candles': df_candles, 'ticks_by_minute': ticks_by_minute,
                'vp': IncrementalVolumeProfile(tick_size=asset['tick_size'], va_percent=asset['va_percent']),
                'state': "INSIDE", 'swing_extreme': 0.0, 'active_trade': None,
                'breakout_time': None, 'breakout_price': None, 'current_session': None,
                'session_start_dt': None,
            }
            print(f"   + {asset['symbol']}: {len(df_candles):,} candles | {len(df_ticks):,} ticks")
    conn.close()
    print(f"   [TIME] Charge en {time.time() - t0:.2f}s")
    if not assets_data:
        print("[ERROR] Aucune donnee chargee.")
        return
    
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
    filtered_by_duration = 0
    filtered_by_poc_strength = 0
    filtered_by_vp_shape = 0
    total_potential_entries = 0
    
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
    active_sessions_display = {}
    for sym, data in assets_data.items():
        sess_list = [s for s, v in data['config'].get('sessions', {}).items() if v]
        active_sessions_display[sym] = sess_list
    
    directions = []
    if any(a['allow_long'] for a in enabled_assets): directions.append("LONG")
    if any(a['allow_short'] for a in enabled_assets): directions.append("SHORT")
    poc_filter_status = "ON" if FILTER_ENTRY_VS_POC else "OFF"
    trailing_status = f"ON (TP1 @ {TP1_RR}R)" if USE_TRAILING else "OFF"
    tp_mode_display = f"R:R {TARGET_RR}" if TP_MODE == "FIXED_RR" else f"POC (min {MIN_RR}R)"
    duration_filter_status = f"ON (max {MAX_BREAKOUT_DURATION_MINUTES}min)" if USE_BREAKOUT_DURATION_FILTER else "OFF"
    vp_structure_status = f"ON (min {MIN_POC_STRENGTH}x)" if USE_VP_STRUCTURE_FILTER else "OFF"
    vp_shape_status = f"ON (exclu: {EXCLUDED_VP_SHAPES})" if USE_VP_SHAPE_FILTER else "OFF"
    vp_reset_status = "PAR SESSION" if RESET_VP_PER_SESSION else "JOURNALIER (23h-0h)"
    
    print(f"\n[BACKTEST] VP Failed Breakout | MULTI-ASSETS: {symbols_list}")
    print(f"[CAPITAL] ${INITIAL_CAPITAL:,.2f} | Risque: {RISK_PERCENT*100}% | TP: {tp_mode_display} | Trailing: {trailing_status}")
    print(f"[CONFIG]  Directions: {directions} | POC Filter: {poc_filter_status}")
    for sym, sess_list in active_sessions_display.items():
        print(f"[CONFIG]  {sym}: Sessions = {sess_list}")
    print(f"[SESSIONS] Heures VP / Heures Trade:")
    for sess_name, cfg in SESSIONS_CONFIG.items():
        vp_h = f"{cfg['vp_start']}-{cfg['vp_end']}"
        trade_h = f"{cfg['trade_start']}-{cfg['trade_end']}"
        print(f"           {sess_name}: VP={vp_h} | Trade={trade_h}")
    print(f"[FILTER]  Breakout Duration: {duration_filter_status}")
    print(f"[FILTER]  VP Structure (POC Strength): {vp_structure_status}")
    print(f"[FILTER]  VP Shape: {vp_shape_status}")
    print(f"[RESET]   VP Reset: {vp_reset_status}")
    
    if DISPLAY_MODE != "NONE":
        print("=" * 120)
        if DISPLAY_MODE == "DAILY":
            print(f"{'DATE':<12} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'PnL %':>7} | {'CAPITAL':>14} | {'DAY DD%':>8} | {'MAX DD%':>8}")
        elif DISPLAY_MODE == "WEEKLY":
            print(f"{'SEMAINE':<12} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'PnL %':>7} | {'CAPITAL':>14} | {'WK DD%':>8} | {'MAX DD%':>8}")
        elif DISPLAY_MODE == "MONTHLY":
            print(f"{'MOIS':<10} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'PnL %':>7} | {'CAPITAL':>14} | {'MTH DD%':>8} | {'MAX DD%':>8}")
        print("-" * 120)
    
    t_start = time.time()
    
    for row in df_all_candles.itertuples():
        symbol = row.symbol
        asset_data = assets_data[symbol]
        config = asset_data['config']
        current_minute = row.dt.floor('T')
        row_day = row.dt.date()
        row_week_start = (row.dt - timedelta(days=row.dt.weekday())).date()
        row_week = row_week_start.strftime('%Y-%m-%d')
        row_month = row.dt.strftime('%Y-%m')
        
        if current_day is not None and row_day != current_day:
            if DISPLAY_MODE == "DAILY" and day_trades > 0:
                day_wr = (day_wins / day_trades * 100) if day_trades > 0 else 0
                day_max_dd_pct = (day_max_dd / day_high_water * 100) if day_high_water > 0 else 0
                day_pnl_pct = ((current_capital - day_start_capital) / day_start_capital * 100) if day_start_capital > 0 else 0
                print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades - day_wins:>4} | {day_wr:>5.1f}% | {day_pnl_r:>+7.1f}R | {day_pnl_pct:>+6.2f}% | ${current_capital:>13,.2f} | {day_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
            day_start_capital = current_capital
            day_trades = 0
            day_wins = 0
            day_pnl_r = 0.0
            day_high_water = current_capital
            day_max_dd = 0.0
        
        if current_week is not None and row_week != current_week:
            if DISPLAY_MODE == "WEEKLY" and week_trades > 0:
                week_wr = (week_wins / week_trades * 100) if week_trades > 0 else 0
                week_max_dd_pct = (week_max_dd / week_high_water * 100) if week_high_water > 0 else 0
                week_pnl_pct = ((current_capital - week_start_capital) / week_start_capital * 100) if week_start_capital > 0 else 0
                print(f"{current_week:<12} | {week_trades:>6} | {week_wins:>4} | {week_trades - week_wins:>4} | {week_wr:>5.1f}% | {week_pnl_r:>+7.1f}R | {week_pnl_pct:>+6.2f}% | ${current_capital:>13,.2f} | {week_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
            week_start_capital = current_capital
            week_trades = 0
            week_wins = 0
            week_pnl_r = 0.0
            week_high_water = current_capital
            week_max_dd = 0.0
        
        if current_month is not None and row_month != current_month:
            if DISPLAY_MODE == "MONTHLY" and month_trades > 0:
                month_wr = (month_wins / month_trades * 100) if month_trades > 0 else 0
                month_max_dd_pct = (month_max_dd / month_high_water * 100) if month_high_water > 0 else 0
                month_pnl_pct = ((current_capital - month_start_capital) / month_start_capital * 100) if month_start_capital > 0 else 0
                print(f"{current_month:<10} | {month_trades:>6} | {month_wins:>4} | {month_trades - month_wins:>4} | {month_wr:>5.1f}% | {month_pnl_r:>+7.1f}R | {month_pnl_pct:>+6.2f}% | ${current_capital:>13,.2f} | {month_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
            month_start_capital = current_capital
            month_trades = 0
            month_wins = 0
            month_pnl_r = 0.0
            month_high_water = current_capital
            month_max_dd = 0.0
        
        current_day = row_day
        current_week = row_week
        current_month = row_month
        
        # =================================================================
        # ÉTAPE 1: SESSION DETECTION & VP UPDATE
        # TOUJOURS exécuté, même avec un trade actif
        # (aligné sur le live qui reconstruit le VP à chaque bougie)
        # =================================================================
        curr_sess = get_session(row.dt)
        asset_sessions = config.get('sessions', {})
        
        if RESET_VP_PER_SESSION:
            session_start = is_session_start(row.dt)
            if session_start and asset_sessions.get(session_start, False):
                asset_data['vp'].reset()
                asset_data['state'] = "INSIDE"
                asset_data['swing_extreme'] = 0.0
                asset_data['current_session'] = session_start
                asset_data['session_start_dt'] = get_session_start_time(session_start, row.dt)
        else:
            if row.dt.hour in [23, 0]:
                asset_data['vp'].reset()
                asset_data['state'] = "INSIDE"
                asset_data['swing_extreme'] = 0.0
        
        # Ajouter les ticks au VP (TOUJOURS, même pendant un trade actif)
        if asset_sessions.get(curr_sess, False):
            if current_minute in asset_data['ticks_by_minute']:
                prices, volumes, timestamps = asset_data['ticks_by_minute'][current_minute]
                session_start_dt = asset_data.get('session_start_dt')
                if session_start_dt is not None:
                    session_start_np = np.datetime64(session_start_dt)
                    mask = timestamps >= session_start_np
                    if mask.any():
                        asset_data['vp'].add_ticks(prices[mask], volumes[mask])
                else:
                    asset_data['vp'].add_ticks(prices, volumes)
        
        # =================================================================
        # ÉTAPE 2: GESTION DES TRADES ACTIFS
        # =================================================================
        active_trade = asset_data['active_trade']
        if active_trade:
            res = None
            if active_trade['type'] == 'SHORT':
                if row.high >= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    if USE_TRAILING and TP_MODE == "POC" and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] - (active_trade['risk'] * TP1_RR)
                        if row.low <= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = TP1_RR * 0.5
                            active_trade['tp1_time'] = row.dt
                    if row.low <= active_trade['tp']:
                        res = "WIN"
                        active_trade['exit_time'] = row.dt
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['tp2_time'] = row.dt
            elif active_trade['type'] == 'LONG':
                if row.low <= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    if USE_TRAILING and TP_MODE == "POC" and not active_trade.get('partial_closed', False):
                        tp1_price = active_trade['entry'] + (active_trade['risk'] * TP1_RR)
                        if row.high >= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = TP1_RR * 0.5
                            active_trade['tp1_time'] = row.dt
                    if row.high >= active_trade['tp']:
                        res = "WIN"
                        active_trade['exit_time'] = row.dt
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['tp2_time'] = row.dt
            
            if res:
                risk_amount = current_capital * RISK_PERCENT
                trade_rr = active_trade['rr']
                partial_pnl_r = active_trade.get('partial_pnl_r', 0)
                if res == "WIN":
                    if USE_TRAILING and TP_MODE == "POC" and active_trade.get('partial_closed', False):
                        pnl_r = partial_pnl_r + (trade_rr * 0.5)
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
                current_capital += pnl
                
                day_trades += 1
                day_pnl_r += pnl_r
                
                week_trades += 1
                week_pnl_r += pnl_r
                if res in ["WIN", "BE"]:
                    week_wins += 1
                
                month_trades += 1
                month_pnl_r += pnl_r
                if res in ["WIN", "BE"]:
                    month_wins += 1
                
                if current_capital > high_water_mark:
                    high_water_mark = current_capital
                if current_capital > day_high_water:
                    day_high_water = current_capital
                if current_capital > week_high_water:
                    week_high_water = current_capital
                if current_capital > month_high_water:
                    month_high_water = current_capital
                
                current_dd = high_water_mark - current_capital
                current_dd_percent = (current_dd / high_water_mark) * 100 if high_water_mark > 0 else 0
                if current_dd > max_dd_amount:
                    max_dd_amount = current_dd
                if current_dd_percent > max_dd_percent:
                    max_dd_percent = current_dd_percent
                
                day_dd = day_high_water - current_capital
                if day_dd > day_max_dd:
                    day_max_dd = day_dd
                
                week_dd = week_high_water - current_capital
                if week_dd > week_max_dd:
                    week_max_dd = week_dd
                
                month_dd = month_high_water - current_capital
                if month_dd > month_max_dd:
                    month_max_dd = month_dd
                
                # Calculer la durée du trade en minutes
                trade_duration_min = 0.0
                if active_trade.get('entry_time') and active_trade.get('exit_time'):
                    trade_duration_min = (active_trade['exit_time'] - active_trade['entry_time']).total_seconds() / 60.0
                
                all_trades.append({
                    'symbol': symbol, 'date': row.dt,
                    'entry_hour': active_trade.get('entry_time').hour if active_trade.get('entry_time') else row.dt.hour,
                    'session': active_trade['session_at_open'], 'type': active_trade['type'],
                    'breakout_time': active_trade.get('breakout_time'), 'breakout_price': active_trade.get('breakout_price'),
                    'breakout_duration_min': active_trade.get('breakout_duration_min'),
                    'poc_strength': active_trade.get('poc_strength'), 'vp_shape': active_trade.get('vp_shape'),
                    'entry_time': active_trade.get('entry_time'), 'entry': active_trade['entry'],
                    'sl': active_trade['original_sl'], 'tp1': active_trade.get('tp1'),
                    'tp1_time': active_trade.get('tp1_time'), 'tp': active_trade['tp'],
                    'tp2_time': active_trade.get('tp2_time'), 'exit_time': active_trade.get('exit_time'),
                    'exit_price': active_trade.get('exit_price'), 'vah_at_entry': active_trade.get('vah_at_entry'),
                    'val_at_entry': active_trade.get('val_at_entry'), 'poc_at_entry': active_trade.get('poc_at_entry'),
                    'rr': active_trade['rr'], 'result': res, 'pnl': pnl, 'pnl_r': pnl_r,
                    'capital_after': current_capital, 'high_water_mark': high_water_mark,
                    'drawdown': high_water_mark - current_capital,
                    'trade_duration_min': trade_duration_min,
                })
                asset_data['active_trade'] = None
            else:
                continue
        
        # =================================================================
        # ÉTAPE 3: STATE MACHINE (seulement si pas de trade actif)
        # =================================================================
        if not asset_sessions.get(curr_sess, False):
            asset_data['state'] = "INSIDE"
            continue
        
        poc, vah, val = asset_data['vp'].get_levels()
        if poc is None:
            continue
        
        poc_strength = asset_data['vp'].get_poc_strength()
        vp_shape = asset_data['vp'].get_profile_shape()
        close, high, low = row.close, row.high, row.low
        state = asset_data['state']
        swing_extreme = asset_data['swing_extreme']
        
        if state == "INSIDE":
            if close > vah:
                asset_data['state'] = "BREAKOUT_UP"
                asset_data['swing_extreme'] = high
                asset_data['breakout_time'] = row.dt
                asset_data['breakout_price'] = close
            elif close < val:
                asset_data['state'] = "BREAKOUT_DOWN"
                asset_data['swing_extreme'] = low
                asset_data['breakout_time'] = row.dt
                asset_data['breakout_price'] = close
        
        elif state == "BREAKOUT_UP":
            asset_data['swing_extreme'] = max(swing_extreme, high)
            if close < vah:
                total_potential_entries += 1
                breakout_duration_min = (row.dt - asset_data['breakout_time']).total_seconds() / 60.0
                duration_ok = True
                if USE_BREAKOUT_DURATION_FILTER:
                    if breakout_duration_min >= MAX_BREAKOUT_DURATION_MINUTES:
                        duration_ok = False
                        filtered_by_duration += 1
                poc_strength_ok = True
                if USE_VP_STRUCTURE_FILTER:
                    if poc_strength is None or poc_strength < MIN_POC_STRENGTH:
                        poc_strength_ok = False
                        filtered_by_poc_strength += 1
                vp_shape_ok = True
                if USE_VP_SHAPE_FILTER:
                    if vp_shape in EXCLUDED_VP_SHAPES:
                        vp_shape_ok = False
                        filtered_by_vp_shape += 1
                if duration_ok and poc_strength_ok and vp_shape_ok and config['allow_short'] and can_trade_now(row.dt, curr_sess):
                    sl_offset = config.get('sl_offset', 0.10)
                    sl = asset_data['swing_extreme'] + sl_offset
                    risk = sl - close
                    if TP_MODE == "POC":
                        tp = poc
                        actual_rr = (close - tp) / risk if risk > 0 else 0
                        if actual_rr > 15:
                            asset_data['state'] = "INSIDE"
                            continue
                    else:
                        tp = close - (risk * TARGET_RR)
                        actual_rr = TARGET_RR
                    poc_ok = (close >= poc) if FILTER_ENTRY_VS_POC else True
                    rr_ok = actual_rr >= MIN_RR
                    if risk > 0 and tp >= val and poc_ok and rr_ok:
                        tp1_price = close - (risk * TP1_RR) if USE_TRAILING and TP_MODE == "POC" else None
                        asset_data['active_trade'] = {
                            'type': 'SHORT', 'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
                            'tp': tp, 'tp1': tp1_price, 'rr': actual_rr, 'session_at_open': curr_sess,
                            'partial_closed': False, 'breakout_time': asset_data['breakout_time'],
                            'breakout_price': asset_data['breakout_price'], 'entry_time': row.dt,
                            'breakout_duration_min': breakout_duration_min, 'poc_strength': poc_strength,
                            'vp_shape': vp_shape, 'tp1_time': None, 'tp2_time': None, 'exit_time': None,
                            'vah_at_entry': vah, 'val_at_entry': val, 'poc_at_entry': poc
                        }
                asset_data['state'] = "INSIDE"
        
        elif state == "BREAKOUT_DOWN":
            asset_data['swing_extreme'] = min(swing_extreme, low)
            if close > val:
                total_potential_entries += 1
                breakout_duration_min = (row.dt - asset_data['breakout_time']).total_seconds() / 60.0
                duration_ok = True
                if USE_BREAKOUT_DURATION_FILTER:
                    if breakout_duration_min >= MAX_BREAKOUT_DURATION_MINUTES:
                        duration_ok = False
                        filtered_by_duration += 1
                poc_strength_ok = True
                if USE_VP_STRUCTURE_FILTER:
                    if poc_strength is None or poc_strength < MIN_POC_STRENGTH:
                        poc_strength_ok = False
                        filtered_by_poc_strength += 1
                vp_shape_ok = True
                if USE_VP_SHAPE_FILTER:
                    if vp_shape in EXCLUDED_VP_SHAPES:
                        vp_shape_ok = False
                        filtered_by_vp_shape += 1
                if duration_ok and poc_strength_ok and vp_shape_ok and config['allow_long'] and can_trade_now(row.dt, curr_sess):
                    sl_offset = config.get('sl_offset', 0.10)
                    sl = asset_data['swing_extreme'] - sl_offset
                    risk = close - sl
                    if TP_MODE == "POC":
                        tp = poc
                        actual_rr = (tp - close) / risk if risk > 0 else 0
                        if actual_rr > 15:
                            asset_data['state'] = "INSIDE"
                            continue
                    else:
                        tp = close + (risk * TARGET_RR)
                        actual_rr = TARGET_RR
                    poc_ok = (close <= poc) if FILTER_ENTRY_VS_POC else True
                    rr_ok = actual_rr >= MIN_RR
                    if risk > 0 and tp <= vah and poc_ok and rr_ok:
                        tp1_price = close + (risk * TP1_RR) if USE_TRAILING and TP_MODE == "POC" else None
                        asset_data['active_trade'] = {
                            'type': 'LONG', 'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
                            'tp': tp, 'tp1': tp1_price, 'rr': actual_rr, 'session_at_open': curr_sess,
                            'partial_closed': False, 'breakout_time': asset_data['breakout_time'],
                            'breakout_price': asset_data['breakout_price'], 'entry_time': row.dt,
                            'breakout_duration_min': breakout_duration_min, 'poc_strength': poc_strength,
                            'vp_shape': vp_shape, 'tp1_time': None, 'tp2_time': None, 'exit_time': None,
                            'vah_at_entry': vah, 'val_at_entry': val, 'poc_at_entry': poc
                        }
                asset_data['state'] = "INSIDE"
    
    if DISPLAY_MODE == "DAILY" and day_trades > 0:
        day_wr = (day_wins / day_trades * 100) if day_trades > 0 else 0
        day_max_dd_pct = (day_max_dd / day_high_water * 100) if day_high_water > 0 else 0
        day_pnl_pct = ((current_capital - day_start_capital) / day_start_capital * 100) if day_start_capital > 0 else 0
        print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades - day_wins:>4} | {day_wr:>5.1f}% | {day_pnl_r:>+7.1f}R | {day_pnl_pct:>+6.2f}% | ${current_capital:>13,.2f} | {day_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
    
    if DISPLAY_MODE == "WEEKLY" and week_trades > 0:
        week_wr = (week_wins / week_trades * 100) if week_trades > 0 else 0
        week_max_dd_pct = (week_max_dd / week_high_water * 100) if week_high_water > 0 else 0
        week_pnl_pct = ((current_capital - week_start_capital) / week_start_capital * 100) if week_start_capital > 0 else 0
        print(f"{current_week:<12} | {week_trades:>6} | {week_wins:>4} | {week_trades - week_wins:>4} | {week_wr:>5.1f}% | {week_pnl_r:>+7.1f}R | {week_pnl_pct:>+6.2f}% | ${current_capital:>13,.2f} | {week_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
    
    if DISPLAY_MODE == "MONTHLY" and month_trades > 0:
        month_wr = (month_wins / month_trades * 100) if month_trades > 0 else 0
        month_max_dd_pct = (month_max_dd / month_high_water * 100) if month_high_water > 0 else 0
        month_pnl_pct = ((current_capital - month_start_capital) / month_start_capital * 100) if month_start_capital > 0 else 0
        print(f"{current_month:<10} | {month_trades:>6} | {month_wins:>4} | {month_trades - month_wins:>4} | {month_wr:>5.1f}% | {month_pnl_r:>+7.1f}R | {month_pnl_pct:>+6.2f}% | ${current_capital:>13,.2f} | {month_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
    
    elapsed = time.time() - t_start
    
    print("\n" + "=" * 120)
    print("[RAPPORT FINAL]")
    print("=" * 120)
    
    if not all_trades:
        print("[ERROR] Aucun trade execute.")
        return
    
    df_trades = pd.DataFrame(all_trades)
    
    # =========================================================================
    # VÉRIFICATION CONFORMITÉ PROP FIRM
    # =========================================================================
    compliance_results = check_prop_firm_rules(df_trades, INITIAL_CAPITAL)
    display_prop_firm_compliance(compliance_results)
    
    # =========================================================================
    # STATISTIQUES CLASSIQUES
    # =========================================================================
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
    avg_loss = abs(df_trades[df_trades['result'] == 'LOSS']['pnl_r'].mean()) if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    df_trades['is_loss'] = df_trades['result'] == 'LOSS'
    df_trades['streak'] = (df_trades['is_loss'] != df_trades['is_loss'].shift()).cumsum()
    losing_streaks = df_trades[df_trades['is_loss']].groupby('streak').size()
    winning_streaks = df_trades[~df_trades['is_loss']].groupby('streak').size()
    max_losing_streak = losing_streaks.max() if len(losing_streaks) > 0 else 0
    max_winning_streak = winning_streaks.max() if len(winning_streaks) > 0 else 0
    recovery_factor = total_pnl / max_dd_amount if max_dd_amount > 0 else float('inf')
    if df_trades['date'].dtype == 'object':
        trading_days = df_trades['date'].nunique()
    else:
        trading_days = pd.to_datetime(df_trades['date']).dt.date.nunique()
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    
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
    avg_rr_all = df_trades['rr'].mean()
    print(f"  Avg R:R (winners):    {avg_rr_winners:>14.2f}")
    print(f"  Avg R:R (all trades): {avg_rr_all:>14.2f}")
    
    print(f"\n{'─' * 60}")
    print("FILTRES APPLIQUES")
    print(f"{'─' * 60}")
    print(f"  Setups potentiels:    {total_potential_entries:>14}")
    print(f"  Filtres (duree):      {filtered_by_duration:>14} ({filtered_by_duration/total_potential_entries*100 if total_potential_entries > 0 else 0:.1f}%)")
    print(f"  Filtres (POC faible): {filtered_by_poc_strength:>14} ({filtered_by_poc_strength/total_potential_entries*100 if total_potential_entries > 0 else 0:.1f}%)")
    print(f"  Filtres (VP Shape):   {filtered_by_vp_shape:>14} ({filtered_by_vp_shape/total_potential_entries*100 if total_potential_entries > 0 else 0:.1f}%)")
    
    # =========================================================================
    # PERFORMANCE PAR DURÉE DE TRADE (PROP FIRM)
    # =========================================================================
    if 'trade_duration_min' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR DURÉE DE TRADE")
        print(f"{'─' * 60}")
        avg_trade_duration = df_trades['trade_duration_min'].mean()
        median_trade_duration = df_trades['trade_duration_min'].median()
        min_trade_duration = df_trades['trade_duration_min'].min()
        max_trade_duration = df_trades['trade_duration_min'].max()
        trades_under_1min = (df_trades['trade_duration_min'] <= 1.0).sum()
        trades_above_1min = (df_trades['trade_duration_min'] > 1.0).sum()
        pct_above_1min = trades_above_1min / total_trades * 100
        
        print(f"  Durée moyenne:          {avg_trade_duration:>10.1f} min")
        print(f"  Durée médiane:          {median_trade_duration:>10.1f} min")
        print(f"  Durée min:              {min_trade_duration:>10.1f} min")
        print(f"  Durée max:              {max_trade_duration:>10.1f} min")
        print(f"  Trades <= 1 min:        {trades_under_1min:>10} ({trades_under_1min/total_trades*100:.1f}%)")
        print(f"  Trades > 1 min:         {trades_above_1min:>10} ({pct_above_1min:.1f}%)")
        
        rule_threshold = PROP_FIRM_RULES.get('min_trade_duration_above_1min_percent', 50.0)
        if pct_above_1min >= rule_threshold:
            print(f"  Prop Firm (>{rule_threshold:.0f}% > 1min): [OK] CONFORME")
        else:
            print(f"  Prop Firm (>{rule_threshold:.0f}% > 1min): [FAIL] NON CONFORME ({pct_above_1min:.1f}% < {rule_threshold:.0f}%)")
        
        print(f"\n  Distribution par durée:")
        duration_buckets = [
            (0, 1, "0-1 min (rapides)"),
            (1, 2, "1-2 min"),
            (2, 5, "2-5 min"),
            (5, 10, "5-10 min"),
            (10, 30, "10-30 min"),
            (30, 60, "30-60 min"),
            (60, 240, "1h-4h"),
            (240, 1440, "4h-24h"),
            (1440, float('inf'), "> 24h"),
        ]
        for lo, hi, label in duration_buckets:
            if hi == float('inf'):
                bucket = df_trades[df_trades['trade_duration_min'] >= lo]
            else:
                bucket = df_trades[(df_trades['trade_duration_min'] >= lo) & (df_trades['trade_duration_min'] < hi)]
            if len(bucket) > 0:
                b_wins = len(bucket[bucket['result'].isin(['WIN', 'BE'])])
                b_wr = b_wins / len(bucket) * 100
                b_pnl = bucket['pnl_r'].sum()
                b_avg = bucket['pnl_r'].mean()
                print(f"     {label:<20}: {len(bucket):>4} trades ({len(bucket)/total_trades*100:>5.1f}%) | WR: {b_wr:>5.1f}% | PnL: {b_pnl:>+8.1f}R | Avg: {b_avg:>+5.2f}R")
    
    if 'poc_strength' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR POC STRENGTH")
        print(f"{'─' * 60}")
        avg_poc_strength = df_trades['poc_strength'].mean()
        median_poc_strength = df_trades['poc_strength'].median()
        print(f"  POC Strength moyen:   {avg_poc_strength:>14.2f}x")
        print(f"  POC Strength median:  {median_poc_strength:>14.2f}x")
        print(f"\n  Performance par niveau de POC Strength:")
        strength_buckets = [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 10.0)]
        for low, high in strength_buckets:
            bucket_trades = df_trades[(df_trades['poc_strength'] >= low) & (df_trades['poc_strength'] < high)]
            if len(bucket_trades) > 0:
                b_wins = len(bucket_trades[bucket_trades['result'].isin(['WIN', 'BE'])])
                b_wr = b_wins / len(bucket_trades) * 100
                b_pnl = bucket_trades['pnl_r'].sum()
                b_avg = bucket_trades['pnl_r'].mean()
                print(f"     {low:.1f}x-{high:.1f}x: {len(bucket_trades):>4} trades | WR: {b_wr:>5.1f}% | PnL: {b_pnl:>+8.1f}R | Avg: {b_avg:>+5.2f}R")
    
    if 'vp_shape' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR FORME DU VP")
        print(f"{'─' * 60}")
        for shape in ["P-SHAPE", "D-SHAPE", "B-SHAPE", "FLAT"]:
            shape_trades = df_trades[df_trades['vp_shape'] == shape]
            if len(shape_trades) > 0:
                s_wins = len(shape_trades[shape_trades['result'].isin(['WIN', 'BE'])])
                s_wr = s_wins / len(shape_trades) * 100
                s_pnl = shape_trades['pnl_r'].sum()
                s_avg = shape_trades['pnl_r'].mean()
                if shape == "P-SHAPE": desc = "(POC haut)  "
                elif shape == "B-SHAPE": desc = "(POC bas)   "
                elif shape == "D-SHAPE": desc = "(POC milieu)"
                else: desc = "(diffus)    "
                print(f"  {shape:<8} {desc} | {len(shape_trades):>4} trades | WR: {s_wr:>5.1f}% | PnL: {s_pnl:>+8.1f}R | Avg: {s_avg:>+5.2f}R")
    
    if 'breakout_duration_min' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR DUREE DE BREAKOUT")
        print(f"{'─' * 60}")
        avg_duration = df_trades['breakout_duration_min'].mean()
        print(f"  Duree moyenne:        {avg_duration:>13.1f} min")
        print(f"\n  Performance par nombre de bougies en breakout:")
        for n_candles in range(1, MAX_BREAKOUT_DURATION_MINUTES + 1):
            low = float(n_candles)
            high = float(n_candles + 1)
            bucket_trades = df_trades[(df_trades['breakout_duration_min'] >= low) & (df_trades['breakout_duration_min'] < high)]
            if len(bucket_trades) > 0:
                b_wins = len(bucket_trades[bucket_trades['result'].isin(['WIN', 'BE'])])
                b_wr = b_wins / len(bucket_trades) * 100
                b_pnl = bucket_trades['pnl_r'].sum()
                b_avg = bucket_trades['pnl_r'].mean()
                print(f"     {n_candles} bougie(s):  {len(bucket_trades):>4} trades | WR: {b_wr:>5.1f}% | PnL: {b_pnl:>+8.1f}R | Avg: {b_avg:>+5.2f}R")
    
    if 'entry_hour' in df_trades.columns:
        print(f"\n{'─' * 60}")
        print("PERFORMANCE PAR HEURE D'ENTREE (UTC)")
        print(f"{'─' * 60}")
        print(f"{'HEURE':<8} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'Avg':>7} | {'PF':>6}")
        print("-" * 75)
        for hour in range(24):
            hour_trades = df_trades[df_trades['entry_hour'] == hour]
            if len(hour_trades) > 0:
                h_total = len(hour_trades)
                h_wins = len(hour_trades[hour_trades['result'].isin(['WIN', 'BE'])])
                h_losses = len(hour_trades[hour_trades['result'] == 'LOSS'])
                h_wr = h_wins / h_total * 100
                h_pnl_r = hour_trades['pnl_r'].sum()
                h_avg = hour_trades['pnl_r'].mean()
                h_gross_profit = hour_trades[hour_trades['pnl'] > 0]['pnl'].sum()
                h_gross_loss = abs(hour_trades[hour_trades['pnl'] < 0]['pnl'].sum())
                h_pf = h_gross_profit / h_gross_loss if h_gross_loss > 0 else float('inf')
                pf_str = f"{h_pf:.2f}" if h_pf != float('inf') else "inf"
                if 0 <= hour < 4: sess_tag = "TKY"
                elif 8 <= hour < 13: sess_tag = "LDN"
                elif 14 <= hour < 21: sess_tag = "NYC"
                else: sess_tag = "OFF"
                print(f"{sess_tag} {hour:02d}:00 | {h_total:>7} | {h_wins:>5} | {h_losses:>5} | {h_wr:>6.1f}% | {h_pnl_r:>+8.1f}R | {h_avg:>+6.2f}R | {pf_str:>6}")
    
    print(f"\n{'─' * 60}")
    print("STATISTIQUES TEMPORELLES")
    print(f"{'─' * 60}")
    print(f"  Jours de Trading:     {trading_days:>14}")
    print(f"  Trades/Jour (avg):    {avg_trades_per_day:>14.1f}")
    print(f"  Max Serie Gagnante:   {max_winning_streak:>14}")
    print(f"  Max Serie Perdante:   {max_losing_streak:>14}")
    print(f"  Temps d'execution:    {elapsed:>13.2f}s")
    
    # Stats sur les jours qualifiants (prop firm)
    df_trades['date_only'] = pd.to_datetime(df_trades['date']).dt.date
    daily_pnl = df_trades.groupby('date_only').agg({'pnl': 'sum', 'capital_after': 'last'}).reset_index()
    daily_pnl['day_start_capital'] = daily_pnl['capital_after'].shift(1).fillna(INITIAL_CAPITAL)
    daily_pnl['day_gain_percent'] = (daily_pnl['pnl'] / daily_pnl['day_start_capital'] * 100)
    min_gain_threshold = PROP_FIRM_RULES.get('min_daily_gain_percent', 0.25)
    qualifying_days = (daily_pnl['day_gain_percent'] >= min_gain_threshold).sum()
    positive_days = (daily_pnl['pnl'] > 0).sum()
    negative_days = (daily_pnl['pnl'] < 0).sum()
    be_days = (daily_pnl['pnl'] == 0).sum()
    
    print(f"\n{'─' * 60}")
    print("STATISTIQUES JOURS (PROP FIRM)")
    print(f"{'─' * 60}")
    print(f"  Jours Qualifiants (>={min_gain_threshold}%): {qualifying_days:>8} / {trading_days} ({qualifying_days/trading_days*100:.1f}%)")
    print(f"  Jours Positifs (>0):   {positive_days:>8} / {trading_days} ({positive_days/trading_days*100:.1f}%)")
    print(f"  Jours Négatifs (<0):   {negative_days:>8} / {trading_days} ({negative_days/trading_days*100:.1f}%)")
    if be_days > 0:
        print(f"  Jours Break Even (=0): {be_days:>8} / {trading_days} ({be_days/trading_days*100:.1f}%)")
    avg_daily_gain = daily_pnl['day_gain_percent'].mean()
    median_daily_gain = daily_pnl['day_gain_percent'].median()
    max_daily_gain = daily_pnl['day_gain_percent'].max()
    min_daily_gain = daily_pnl['day_gain_percent'].min()
    print(f"  Gain journalier moyen: {avg_daily_gain:>13.2f}%")
    print(f"  Gain journalier médian:{median_daily_gain:>13.2f}%")
    print(f"  Meilleur jour:         {max_daily_gain:>13.2f}%")
    print(f"  Pire jour:             {min_daily_gain:>13.2f}%")
    
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR ASSET")
    print(f"{'─' * 60}")
    print(f"{'ASSET':<15} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6}")
    print("-" * 80)
    for symbol in symbols_list:
        sym_trades = df_trades[df_trades['symbol'] == symbol]
        if len(sym_trades) == 0: continue
        s_total = len(sym_trades)
        s_wins = len(sym_trades[sym_trades['result'].isin(['WIN', 'BE'])])
        s_losses = len(sym_trades[sym_trades['result'] == 'LOSS'])
        s_wr = s_wins / s_total * 100
        s_pnl_r = sym_trades['pnl_r'].sum()
        s_gross_profit = sym_trades[sym_trades['pnl'] > 0]['pnl'].sum()
        s_gross_loss = abs(sym_trades[sym_trades['pnl'] < 0]['pnl'].sum())
        s_pf = s_gross_profit / s_gross_loss if s_gross_loss > 0 else float('inf')
        pf_str = f"{s_pf:.2f}" if s_pf != float('inf') else "inf"
        print(f"{symbol:<15} | {s_total:>7} | {s_wins:>5} | {s_losses:>5} | {s_wr:>6.1f}% | {s_pnl_r:>+8.1f}R | {pf_str:>6}")
    
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR SESSION")
    print(f"{'─' * 60}")
    print(f"{'SESSION':<10} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6}")
    print("-" * 75)
    for session in ["TOKYO", "LONDON", "NY"]:
        sess_trades = df_trades[df_trades['session'] == session]
        if len(sess_trades) == 0: continue
        s_total = len(sess_trades)
        s_wins = len(sess_trades[sess_trades['result'].isin(['WIN', 'BE'])])
        s_losses = len(sess_trades[sess_trades['result'] == 'LOSS'])
        s_wr = s_wins / s_total * 100
        s_pnl_r = sess_trades['pnl_r'].sum()
        s_gross_profit = sess_trades[sess_trades['pnl'] > 0]['pnl'].sum()
        s_gross_loss = abs(sess_trades[sess_trades['pnl'] < 0]['pnl'].sum())
        s_pf = s_gross_profit / s_gross_loss if s_gross_loss > 0 else float('inf')
        pf_str = f"{s_pf:.2f}" if s_pf != float('inf') else "inf"
        print(f"{session:<10} | {s_total:>7} | {s_wins:>5} | {s_losses:>5} | {s_wr:>6.1f}% | {s_pnl_r:>+8.1f}R | {pf_str:>6}")
    
    print(f"\n{'─' * 60}")
    print("BREAKDOWN PAR DIRECTION")
    print(f"{'─' * 60}")
    for trade_type in ["LONG", "SHORT"]:
        type_trades = df_trades[df_trades['type'] == trade_type]
        if len(type_trades) == 0: continue
        t_total = len(type_trades)
        t_wins = len(type_trades[type_trades['result'].isin(['WIN', 'BE'])])
        t_losses = len(type_trades[type_trades['result'] == 'LOSS'])
        t_wr = t_wins / t_total * 100
        t_pnl_r = type_trades['pnl_r'].sum()
        t_gross_profit = type_trades[type_trades['pnl'] > 0]['pnl'].sum()
        t_gross_loss = abs(type_trades[type_trades['pnl'] < 0]['pnl'].sum())
        t_pf = t_gross_profit / t_gross_loss if t_gross_loss > 0 else float('inf')
        pf_str = f"{t_pf:.2f}" if t_pf != float('inf') else "inf"
        print(f"  {trade_type:<8} | {t_total:>7} trades | {t_wins:>5} W | {t_losses:>5} L | {t_wr:>5.1f}% | {t_pnl_r:>+8.1f}R | PF {pf_str}")
    
    if SHOW_OPEN_TRADES:
        display_open_trades(assets_data, current_capital)
    
    if SHOW_LAST_TRADES:
        display_last_trades(df_trades, LAST_TRADES_COUNT)
    
    print("=" * 120)


if __name__ == "__main__":
    run_backtest()