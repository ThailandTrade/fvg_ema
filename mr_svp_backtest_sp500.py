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

# --- CONFIGURATION ---
SYMBOL = "US500.cash"
CANDLE_TABLE = "candles_mt5_us500_cash_1m"
TICK_TABLE = "market_ticks_us500"
TICK_SIZE = 0.01
VA_PERCENT = 0.70

# MODE DE TARGET
TP_MODE = "POC"  # "FIXED_RR" = R:R fixe | "POC" = Target au POC
TARGET_RR = 3.0       # Utilisé uniquement si TP_MODE = "FIXED_RR"
MIN_RR = 1.0          # R:R minimum accepté (pour mode POC, évite les trades à faible potentiel)

# TRAILING (mode POC uniquement)
USE_TRAILING = True  # True = TP1 à MIN_RR (50%), TP2 au POC (50%) + SL→BE | False = 100% au target

# GESTION DU CAPITAL
INITIAL_CAPITAL = 50000.0
RISK_PERCENT = 0.004  # 0.1%

# FILTRES DE DIRECTION
ALLOW_LONG = True
ALLOW_SHORT = False

# FILTRES DE QUALITÉ
FILTER_ENTRY_VS_POC = True  # False = désactiver le filtre "entrée du bon côté du POC"

# FILTRES DE SESSION
USE_TOKYO = True
USE_LONDON = False
USE_NY = False

# DATE DE DÉBUT
START_DATE_STR = "2025-07-01 00:00:00"

# --- DB CONNECTION ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('PG_HOST'), port=os.getenv('PG_PORT'),
            database=os.getenv('PG_DB'), user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"❌ Erreur DB: {e}")
        sys.exit(1)


# =============================================================================
# CLASSE VP INCRÉMENTAL
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
            self._cache_valid = True
            return
        
        sorted_bins = sorted(self.profile.keys())
        volumes = np.array([self.profile[b] for b in sorted_bins])
        poc_idx = np.argmax(volumes)
        self._cached_poc = sorted_bins[poc_idx]
        
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


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
def load_all_data(conn):
    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = requested_start - timedelta(hours=2)
    
    print("📊 Chargement des données...")
    t0 = time.time()
    
    ts_start = int(data_start.timestamp() * 1000)
    query_candles = f"""
        SELECT ts, open, high, low, close 
        FROM {CANDLE_TABLE}
        WHERE ts >= {ts_start}
        ORDER BY ts ASC
    """
    df_candles = pd.read_sql(query_candles, conn)
    df_candles['dt'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)
    
    t_start = data_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = df_candles['dt'].max().strftime("%Y-%m-%d %H:%M:%S")
    
    query_ticks = f"""
        SELECT time, last as price, volume 
        FROM {TICK_TABLE}
        WHERE symbol = '{SYMBOL}' 
          AND time >= '{t_start}' 
          AND time <= '{t_end}'
        ORDER BY time ASC
    """
    df_ticks = pd.read_sql(query_ticks, conn)
    df_ticks['time'] = pd.to_datetime(df_ticks['time'], utc=True)
    
    print(f"   ✓ {len(df_candles):,} candles | {len(df_ticks):,} ticks | {time.time() - t0:.2f}s")
    
    return df_candles, df_ticks


def get_session(dt):
    h = dt.hour + dt.minute / 60.0
    if 0 <= h < 8:
        return "TOKYO"
    if 8 <= h < 14.5:
        return "LONDON"
    if 14.5 <= h < 21:
        return "NY"
    return "AUTRE"


# =============================================================================
# MOTEUR DE BACKTEST
# =============================================================================
def run_backtest():
    conn = get_db_connection()
    df_candles, df_ticks = load_all_data(conn)
    conn.close()
    
    if df_candles.empty:
        print("❌ Aucune donnée.")
        return
    
    # Index des ticks par minute
    df_ticks['minute'] = df_ticks['time'].dt.floor('T')
    ticks_by_minute = df_ticks.groupby('minute').apply(
        lambda g: (g['price'].values, g['volume'].values)
    ).to_dict()
    
    # --- INITIALISATION ---
    vp = IncrementalVolumeProfile(tick_size=TICK_SIZE, va_percent=VA_PERCENT)
    sessions_config = {"TOKYO": USE_TOKYO, "LONDON": USE_LONDON, "NY": USE_NY, "AUTRE": False}
    
    state = "INSIDE"
    swing_extreme = 0.0
    active_trade = None
    session_start_dt = df_candles.iloc[0]['dt']
    breakout_time = None
    breakout_price = None
    
    # --- MÉTRIQUES GLOBALES ---
    current_capital = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    max_dd_amount = 0.0
    all_trades = []  # Liste de tous les trades pour analyse
    
    # --- MÉTRIQUES JOURNALIÈRES ---
    current_day = None
    day_start_capital = INITIAL_CAPITAL
    day_trades = 0
    day_wins = 0
    day_pnl_r = 0.0  # PnL en R
    day_high_water = INITIAL_CAPITAL
    day_max_dd = 0.0
    
    # --- AFFICHAGE ---
    active_sessions = [s for s, v in sessions_config.items() if v]
    directions = []
    if ALLOW_LONG: directions.append("LONG")
    if ALLOW_SHORT: directions.append("SHORT")
    poc_filter_status = "ON" if FILTER_ENTRY_VS_POC else "OFF"
    trailing_status = "ON" if USE_TRAILING else "OFF"
    tp_mode_display = f"R:R {TARGET_RR}" if TP_MODE == "FIXED_RR" else f"POC (min {MIN_RR}R)"
    print(f"\n🚀 Backtest VP Failed Breakout | {SYMBOL}")
    print(f"💰 Capital: ${INITIAL_CAPITAL:,.2f} | Risque: {RISK_PERCENT*100}% | TP: {tp_mode_display} | Trailing: {trailing_status}")
    print(f"📅 Sessions: {active_sessions} | Directions: {directions} | POC Filter: {poc_filter_status}")
    print("=" * 100)
    print(f"{'DATE':<12} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'CAPITAL':>12} | {'DAY DD':>8} | {'MAX DD':>8}")
    print("-" * 100)
    
    t_start = time.time()
    
    # --- BOUCLE PRINCIPALE ---
    for row in df_candles.itertuples():
        current_minute = row.dt.floor('T')
        row_day = row.dt.date()
        
        # CHANGEMENT DE JOUR - Afficher le résumé
        if current_day is not None and row_day != current_day:
            day_wr = (day_wins / day_trades * 100) if day_trades > 0 else 0
            day_change = current_capital - day_start_capital
            day_change_sign = "+" if day_change >= 0 else ""
            print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades - day_wins:>4} | {day_wr:>5.1f}% | {day_pnl_r:>+7.1f}R | ${current_capital:>11,.2f} | -${day_max_dd:>7.2f} | -${max_dd_amount:>7.2f}")
            
            # Reset métriques journalières
            day_start_capital = current_capital
            day_trades = 0
            day_wins = 0
            day_pnl_r = 0.0
            day_high_water = current_capital
            day_max_dd = 0.0
        
        current_day = row_day
        
        # 1. GESTION DU TRADE ACTIF
        if active_trade:
            res = None
            partial_hit = False
            
            if active_trade['type'] == 'SHORT':
                # Vérifier SL
                if row.high >= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    # Mode Trailing : vérifier TP1 (partial) puis TP2 (final)
                    if USE_TRAILING and TP_MODE == "POC" and not active_trade.get('partial_closed', False):
                        # TP1 = prix à MIN_RR
                        tp1_price = active_trade['entry'] - (active_trade['risk'] * MIN_RR)
                        if row.low <= tp1_price:
                            # TP1 touché - fermer 50%, move SL to BE
                            partial_hit = True
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']  # SL → Breakeven
                            active_trade['partial_pnl_r'] = MIN_RR * 0.5  # 50% de la pos à MIN_RR
                            active_trade['tp1_time'] = row.dt
                    
                    # Vérifier TP final (POC ou FIXED)
                    if row.low <= active_trade['tp']:
                        res = "WIN"
                        active_trade['exit_time'] = row.dt
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['tp2_time'] = row.dt
            
            elif active_trade['type'] == 'LONG':
                # Vérifier SL
                if row.low <= active_trade['sl']:
                    res = "LOSS"
                    active_trade['exit_time'] = row.dt
                    active_trade['exit_price'] = active_trade['sl']
                else:
                    # Mode Trailing : vérifier TP1 (partial) puis TP2 (final)
                    if USE_TRAILING and TP_MODE == "POC" and not active_trade.get('partial_closed', False):
                        # TP1 = prix à MIN_RR
                        tp1_price = active_trade['entry'] + (active_trade['risk'] * MIN_RR)
                        if row.high >= tp1_price:
                            # TP1 touché - fermer 50%, move SL to BE
                            partial_hit = True
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']  # SL → Breakeven
                            active_trade['partial_pnl_r'] = MIN_RR * 0.5  # 50% de la pos à MIN_RR
                            active_trade['tp1_time'] = row.dt
                    
                    # Vérifier TP final (POC ou FIXED)
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
                        # 50% déjà pris à MIN_RR + 50% au POC
                        pnl_r = partial_pnl_r + (trade_rr * 0.5)
                    else:
                        pnl_r = trade_rr
                    pnl = risk_amount * pnl_r
                    day_wins += 1
                else:  # LOSS
                    if USE_TRAILING and active_trade.get('partial_closed', False):
                        # SL touché après partial = BE sur la 2ème moitié
                        pnl_r = partial_pnl_r  # On garde juste le gain du partial
                        pnl = risk_amount * pnl_r
                        res = "BE"  # Breakeven (partial win)
                        active_trade['exit_price'] = active_trade['entry']  # Sorti au BE
                        day_wins += 1  # Compté comme win car positif
                    else:
                        pnl = -risk_amount
                        pnl_r = -1.0
                
                current_capital += pnl
                day_trades += 1
                day_pnl_r += pnl_r
                
                # Update high water marks
                if current_capital > high_water_mark:
                    high_water_mark = current_capital
                if current_capital > day_high_water:
                    day_high_water = current_capital
                
                # Update drawdowns
                current_dd = high_water_mark - current_capital
                if current_dd > max_dd_amount:
                    max_dd_amount = current_dd
                
                day_dd = day_high_water - current_capital
                if day_dd > day_max_dd:
                    day_max_dd = day_dd
                
                # Enregistrer le trade avec tous les détails
                all_trades.append({
                    'date': row.dt,
                    'session': active_trade['session_at_open'],
                    'type': active_trade['type'],
                    'breakout_time': active_trade.get('breakout_time'),
                    'breakout_price': active_trade.get('breakout_price'),
                    'entry_time': active_trade.get('entry_time'),
                    'entry': active_trade['entry'],
                    'sl': active_trade['original_sl'],
                    'tp1': active_trade.get('tp1'),
                    'tp1_time': active_trade.get('tp1_time'),
                    'tp': active_trade['tp'],
                    'tp2_time': active_trade.get('tp2_time'),
                    'exit_time': active_trade.get('exit_time'),
                    'exit_price': active_trade.get('exit_price'),
                    'vah_at_entry': active_trade.get('vah_at_entry'),
                    'val_at_entry': active_trade.get('val_at_entry'),
                    'poc_at_entry': active_trade.get('poc_at_entry'),
                    'rr': active_trade['rr'],
                    'result': res,
                    'pnl': pnl,
                    'pnl_r': pnl_r,
                    'capital_after': current_capital,
                    'high_water_mark': high_water_mark,
                    'drawdown': high_water_mark - current_capital
                })
                
                active_trade = None
            else:
                continue
        
        # 2. ZONE DE RESET (23h00 - 01h00 UTC)
        if row.dt.hour in [23, 0]:
            if row.dt.hour == 23:
                session_start_dt = row.dt.replace(minute=0, second=0, microsecond=0)
            else:
                session_start_dt = (row.dt - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
            vp.reset()
            state = "INSIDE"
            swing_extreme = 0.0
            continue
        
        # 3. FILTRE DE SESSION
        curr_sess = get_session(row.dt)
        if not sessions_config.get(curr_sess, False):
            state = "INSIDE"
            continue
        
        # 4. MISE À JOUR VP
        if current_minute in ticks_by_minute:
            prices, volumes = ticks_by_minute[current_minute]
            vp.add_ticks(prices, volumes)
        
        poc, vah, val = vp.get_levels()
        if poc is None:
            continue
        
        close, high, low = row.close, row.high, row.low
        
        # 5. LOGIQUE STRATÉGIE
        if state == "INSIDE":
            if close > vah:
                state = "BREAKOUT_UP"
                swing_extreme = high
                breakout_time = row.dt
                breakout_price = close
            elif close < val:
                state = "BREAKOUT_DOWN"
                swing_extreme = low
                breakout_time = row.dt
                breakout_price = close
        
        elif state == "BREAKOUT_UP":
            swing_extreme = max(swing_extreme, high)
            if close < vah:
                if ALLOW_SHORT:
                    sl = swing_extreme + 0.10
                    risk = sl - close
                    
                    # Calcul du TP selon le mode
                    if TP_MODE == "POC":
                        tp = poc
                        actual_rr = (close - tp) / risk if risk > 0 else 0
                    else:  # FIXED_RR
                        tp = close - (risk * TARGET_RR)
                        actual_rr = TARGET_RR
                    
                    poc_ok = (close >= poc) if FILTER_ENTRY_VS_POC else True
                    rr_ok = actual_rr >= MIN_RR
                    
                    if risk > 0 and tp >= val and poc_ok and rr_ok:
                        # Calculer TP1 pour trailing
                        tp1_price = close - (risk * MIN_RR) if USE_TRAILING and TP_MODE == "POC" else None
                        
                        active_trade = {
                            'type': 'SHORT', 
                            'entry': close, 
                            'sl': sl,
                            'original_sl': sl,
                            'risk': risk,
                            'tp': tp,
                            'tp1': tp1_price,
                            'rr': actual_rr,
                            'session_at_open': curr_sess,
                            'partial_closed': False,
                            'breakout_time': breakout_time,
                            'breakout_price': breakout_price,
                            'entry_time': row.dt,
                            'tp1_time': None,
                            'tp2_time': None,
                            'exit_time': None,
                            'vah_at_entry': vah,
                            'val_at_entry': val,
                            'poc_at_entry': poc
                        }
                state = "INSIDE"
        
        elif state == "BREAKOUT_DOWN":
            swing_extreme = min(swing_extreme, low)
            if close > val:
                if ALLOW_LONG:
                    sl = swing_extreme - 0.10
                    risk = close - sl
                    
                    # Calcul du TP selon le mode
                    if TP_MODE == "POC":
                        tp = poc
                        actual_rr = (tp - close) / risk if risk > 0 else 0
                    else:  # FIXED_RR
                        tp = close + (risk * TARGET_RR)
                        actual_rr = TARGET_RR
                    
                    poc_ok = (close <= poc) if FILTER_ENTRY_VS_POC else True
                    rr_ok = actual_rr >= MIN_RR
                    
                    if risk > 0 and tp <= vah and poc_ok and rr_ok:
                        # Calculer TP1 pour trailing
                        tp1_price = close + (risk * MIN_RR) if USE_TRAILING and TP_MODE == "POC" else None
                        
                        active_trade = {
                            'type': 'LONG', 
                            'entry': close, 
                            'sl': sl,
                            'original_sl': sl,
                            'risk': risk,
                            'tp': tp,
                            'tp1': tp1_price,
                            'rr': actual_rr,
                            'session_at_open': curr_sess,
                            'partial_closed': False,
                            'breakout_time': breakout_time,
                            'breakout_price': breakout_price,
                            'entry_time': row.dt,
                            'tp1_time': None,
                            'tp2_time': None,
                            'exit_time': None,
                            'vah_at_entry': vah,
                            'val_at_entry': val,
                            'poc_at_entry': poc
                        }
                state = "INSIDE"
    
    # Afficher le dernier jour
    if day_trades > 0:
        day_wr = (day_wins / day_trades * 100) if day_trades > 0 else 0
        print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades - day_wins:>4} | {day_wr:>5.1f}% | {day_pnl_r:>+7.1f}R | ${current_capital:>11,.2f} | -${day_max_dd:>7.2f} | -${max_dd_amount:>7.2f}")
    
    elapsed = time.time() - t_start
    
    # ==========================================================================
    # RAPPORT FINAL DÉTAILLÉ
    # ==========================================================================
    print("\n" + "=" * 100)
    print("📊 RAPPORT FINAL")
    print("=" * 100)
    
    if not all_trades:
        print("❌ Aucun trade exécuté.")
        return
    
    df_trades = pd.DataFrame(all_trades)
    
    # --- MÉTRIQUES GLOBALES ---
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
    
    # Expectancy (WIN et BE sont des résultats positifs)
    positive_trades = df_trades[df_trades['result'].isin(['WIN', 'BE'])]
    avg_win = positive_trades['pnl_r'].mean() if len(positive_trades) > 0 else 0
    avg_loss = abs(df_trades[df_trades['result'] == 'LOSS']['pnl_r'].mean()) if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    
    # Séries (BE compte comme non-perte)
    df_trades['is_loss'] = df_trades['result'] == 'LOSS'
    df_trades['streak'] = (df_trades['is_loss'] != df_trades['is_loss'].shift()).cumsum()
    losing_streaks = df_trades[df_trades['is_loss']].groupby('streak').size()
    winning_streaks = df_trades[~df_trades['is_loss']].groupby('streak').size()
    max_losing_streak = losing_streaks.max() if len(losing_streaks) > 0 else 0
    max_winning_streak = winning_streaks.max() if len(winning_streaks) > 0 else 0
    
    # Recovery Factor
    recovery_factor = total_pnl / max_dd_amount if max_dd_amount > 0 else float('inf')
    
    # Jours de trading
    trading_days = df_trades['date'].dt.date.nunique()
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    
    print(f"\n{'─' * 50}")
    print("📈 PERFORMANCE GLOBALE")
    print(f"{'─' * 50}")
    print(f"  Capital Initial:      ${INITIAL_CAPITAL:>14,.2f}")
    print(f"  Capital Final:        ${current_capital:>14,.2f}")
    print(f"  Profit/Perte:         ${total_pnl:>+14,.2f} ({total_pnl/INITIAL_CAPITAL*100:+.2f}%)")
    print(f"  Gain Total:           {total_pnl_r:>+14.1f} R")
    print(f"  Max Drawdown:         ${max_dd_amount:>14,.2f} ({max_dd_amount/high_water_mark*100:.2f}%)")
    print(f"  Recovery Factor:      {recovery_factor:>14.2f}")
    
    print(f"\n{'─' * 50}")
    print("🎯 STATISTIQUES DE TRADING")
    print(f"{'─' * 50}")
    print(f"  Trades Total:         {total_trades:>14}")
    if be_trades > 0:
        print(f"  Wins / BE / Losses:   {wins:>5} / {be_trades:>3} / {losses:<5}")
    else:
        print(f"  Wins / Losses:        {wins:>6} / {losses:<6}")
    print(f"  Win Rate:             {win_rate:>13.2f}%")
    print(f"  Profit Factor:        {profit_factor:>14.2f}")
    print(f"  Expectancy:           {expectancy:>+13.2f} R")
    print(f"  Avg PnL/Trade:        {avg_pnl_r:>+13.2f} R")
    
    # R:R moyen (utile pour mode POC)
    avg_rr_winners = df_trades[df_trades['result'] == 'WIN']['rr'].mean() if wins > 0 else 0
    avg_rr_all = df_trades['rr'].mean()
    print(f"  Avg R:R (winners):    {avg_rr_winners:>14.2f}")
    print(f"  Avg R:R (all trades): {avg_rr_all:>14.2f}")
    
    print(f"\n{'─' * 50}")
    print("📅 STATISTIQUES TEMPORELLES")
    print(f"{'─' * 50}")
    print(f"  Jours de Trading:     {trading_days:>14}")
    print(f"  Trades/Jour (avg):    {avg_trades_per_day:>14.1f}")
    print(f"  Max Série Gagnante:   {max_winning_streak:>14}")
    print(f"  Max Série Perdante:   {max_losing_streak:>14}")
    print(f"  Temps d'exécution:    {elapsed:>13.2f}s")
    
    # --- BREAKDOWN PAR SESSION ---
    print(f"\n{'─' * 50}")
    print("🌍 BREAKDOWN PAR SESSION")
    print(f"{'─' * 50}")
    print(f"{'SESSION':<10} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6} | {'EXP':>7} | {'MAX DD%':>8}")
    print("-" * 95)
    
    for session in ["TOKYO", "LONDON", "NY"]:
        status = "ON" if sessions_config[session] else "OFF"
        sess_trades = df_trades[df_trades['session'] == session]
        
        if len(sess_trades) == 0:
            print(f"{session:<10} | {'--':>7} | {'--':>5} | {'--':>5} | {'--':>7} | {'--':>9} | {'--':>6} | {'--':>7} | {'--':>8}  ({status})")
            continue
        
        s_total = len(sess_trades)
        s_wins = len(sess_trades[sess_trades['result'] == 'WIN'])
        s_losses = s_total - s_wins
        s_wr = s_wins / s_total * 100
        s_pnl_r = sess_trades['pnl_r'].sum()
        
        s_gross_profit = sess_trades[sess_trades['pnl'] > 0]['pnl'].sum()
        s_gross_loss = abs(sess_trades[sess_trades['pnl'] < 0]['pnl'].sum())
        s_pf = s_gross_profit / s_gross_loss if s_gross_loss > 0 else float('inf')
        
        s_avg_win = sess_trades[sess_trades['result'] == 'WIN']['pnl_r'].mean() if s_wins > 0 else 0
        s_avg_loss = abs(sess_trades[sess_trades['result'] == 'LOSS']['pnl_r'].mean()) if s_losses > 0 else 0
        s_exp = (s_wr/100 * s_avg_win) - ((1 - s_wr/100) * s_avg_loss)
        
        # Calcul du Max DD % pour cette session
        s_max_dd = sess_trades['drawdown'].max()
        s_max_dd_pct = (s_max_dd / INITIAL_CAPITAL) * 100
        
        print(f"{session:<10} | {s_total:>7} | {s_wins:>5} | {s_losses:>5} | {s_wr:>6.1f}% | {s_pnl_r:>+8.1f}R | {s_pf:>6.2f} | {s_exp:>+6.2f}R | {s_max_dd_pct:>7.2f}%  ({status})")
    
    print("-" * 95)
    max_dd_pct = (max_dd_amount / INITIAL_CAPITAL) * 100
    print(f"{'TOTAL':<10} | {total_trades:>7} | {wins:>5} | {losses:>5} | {win_rate:>6.1f}% | {total_pnl_r:>+8.1f}R | {profit_factor:>6.2f} | {expectancy:>+6.2f}R | {max_dd_pct:>7.2f}%")
    
    # --- BREAKDOWN PAR TYPE DE TRADE ---
    print(f"\n{'─' * 50}")
    print("📊 BREAKDOWN PAR DIRECTION")
    print(f"{'─' * 50}")
    print(f"{'TYPE':<10} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6}")
    print("-" * 65)
    
    for trade_type in ["LONG", "SHORT"]:
        type_trades = df_trades[df_trades['type'] == trade_type]
        if len(type_trades) == 0:
            continue
        
        t_total = len(type_trades)
        t_wins = len(type_trades[type_trades['result'] == 'WIN'])
        t_losses = t_total - t_wins
        t_wr = t_wins / t_total * 100
        t_pnl_r = type_trades['pnl_r'].sum()
        
        t_gross_profit = type_trades[type_trades['pnl'] > 0]['pnl'].sum()
        t_gross_loss = abs(type_trades[type_trades['pnl'] < 0]['pnl'].sum())
        t_pf = t_gross_profit / t_gross_loss if t_gross_loss > 0 else float('inf')
        
        emoji = "🟢" if trade_type == "LONG" else "🔴"
        print(f"{emoji} {trade_type:<8} | {t_total:>7} | {t_wins:>5} | {t_losses:>5} | {t_wr:>6.1f}% | {t_pnl_r:>+8.1f}R | {t_pf:>6.2f}")
    
    print("=" * 100)
    
    # # ==========================================================================
    # # 10 DERNIERS TRADES - VALIDATION MANUELLE
    # # ==========================================================================
    # print(f"\n{'─' * 100}")
    # print("🔍 10 DERNIERS TRADES (Validation Manuelle)")
    # print(f"{'─' * 100}")
    
    # last_10 = df_trades.tail(10)
    
    # for idx, trade in last_10.iterrows():
        # # Formater les timestamps
        # def fmt_time(t):
            # return t.strftime('%d/%m/%Y %H:%M') if pd.notna(t) and t is not None else "N/A"
        
        # # Recalculer le risk depuis SL
        # if trade['type'] == 'LONG':
            # risk_calc = trade['entry'] - trade['sl']
        # else:
            # risk_calc = trade['sl'] - trade['entry']
        
        # # Emoji résultat
        # if trade['result'] == 'WIN':
            # res_emoji = "✅ WIN "
        # elif trade['result'] == 'BE':
            # res_emoji = "🟡 BE  "
        # else:
            # res_emoji = "❌ LOSS"
        
        # # Direction emoji
        # dir_emoji = "🟢 LONG " if trade['type'] == 'LONG' else "🔴 SHORT"
        
        # print(f"\n  ╔{'═' * 90}╗")
        # print(f"  ║ Trade #{idx + 1:03d} | {dir_emoji} | Session: {trade['session']:<6} | Résultat: {res_emoji} | PnL: {trade['pnl_r']:+.2f}R")
        # print(f"  ╠{'═' * 90}╣")
        
        # # Niveaux VP au moment de l'entrée
        # vah = trade.get('vah_at_entry', 'N/A')
        # val = trade.get('val_at_entry', 'N/A')
        # poc = trade.get('poc_at_entry', 'N/A')
        # vah_str = f"{vah:.2f}" if isinstance(vah, (int, float)) else vah
        # val_str = f"{val:.2f}" if isinstance(val, (int, float)) else val
        # poc_str = f"{poc:.2f}" if isinstance(poc, (int, float)) else poc
        
        # print(f"  ║ 📊 VP LEVELS:  VAH: {vah_str}  |  POC: {poc_str}  |  VAL: {val_str}")
        # print(f"  ╠{'─' * 90}╣")
        
        # # Timeline du trade
        # print(f"  ║ ⏱️  TIMELINE:")
        
        # breakout_time = trade.get('breakout_time')
        # breakout_price = trade.get('breakout_price')
        # print(f"  ║    1️⃣  BREAKOUT:      {fmt_time(breakout_time):<20} | Prix: {breakout_price if breakout_price else 'N/A'}")
        
        # entry_time = trade.get('entry_time')
        # print(f"  ║    2️⃣  ENTRY:         {fmt_time(entry_time):<20} | Prix: {trade['entry']:.2f}")
        
        # if USE_TRAILING and TP_MODE == "POC":
            # tp1 = trade.get('tp1')
            # tp1_time = trade.get('tp1_time')
            # tp1_str = f"{tp1:.2f}" if tp1 else "N/A"
            # tp1_status = "✅ Touché" if tp1_time else "❌ Non touché"
            # print(f"  ║    3️⃣  TP1 ({MIN_RR}R):      {fmt_time(tp1_time):<20} | Prix: {tp1_str:<10} | {tp1_status}")
            
            # tp2_time = trade.get('tp2_time')
            # tp2_status = "✅ Touché" if tp2_time else "❌ Non touché"
            # print(f"  ║    4️⃣  TP2 (POC):     {fmt_time(tp2_time):<20} | Prix: {trade['tp']:.2f}       | {tp2_status}")
        # else:
            # print(f"  ║    3️⃣  TP:            {fmt_time(trade.get('tp2_time')):<20} | Prix: {trade['tp']:.2f}")
        
        # exit_time = trade.get('exit_time')
        # exit_price = trade.get('exit_price')
        # exit_price_str = f"{exit_price:.2f}" if exit_price else "N/A"
        # print(f"  ║    🏁 EXIT:          {fmt_time(exit_time):<20} | Prix: {exit_price_str}")
        
        # print(f"  ╠{'─' * 90}╣")
        
        # # Niveaux du trade
        # print(f"  ║ 🎯 TRADE LEVELS:")
        # print(f"  ║    Entry:  {trade['entry']:.2f}")
        # print(f"  ║    SL:     {trade['sl']:.2f}  (Risk: {risk_calc:.2f})")
        # if USE_TRAILING and TP_MODE == "POC" and trade.get('tp1'):
            # print(f"  ║    TP1:    {trade['tp1']:.2f}  (à {MIN_RR}R = 50% position)")
            # print(f"  ║    TP2:    {trade['tp']:.2f}  (POC = 50% restant)")
        # else:
            # print(f"  ║    TP:     {trade['tp']:.2f}")
        # print(f"  ║    R:R:    {trade['rr']:.2f}")
        
        # print(f"  ╠{'─' * 90}╣")
        # print(f"  ║ 💰 Capital après: ${trade['capital_after']:,.2f}")
        # print(f"  ╚{'═' * 90}╝")
    
    # print(f"\n{'─' * 100}")
    # print("✅ Fin du rapport - Copiez les timestamps ci-dessus pour vérifier sur TradingView")
    # print("=" * 100)


if __name__ == "__main__":
    run_backtest()