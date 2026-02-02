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

# Configuration globale
START_DATE_STR = "2025-01-01 00:00:00"
INITIAL_CAPITAL = 50000.0
RISK_PERCENT = 0.003  # 0.5% par trade

# Mode de target
TP_MODE = "POC"  # "FIXED_RR" = R:R fixe | "POC" = Target au POC
TARGET_RR = 3.0
MIN_RR = 1.3

# Trailing
USE_TRAILING = True

# Filtres globaux
FILTER_ENTRY_VS_POC = True
USE_TOKYO = True
USE_LONDON = False
USE_NY = False

# =============================================================================
# CONFIGURATION PAR ASSET
# =============================================================================
ASSETS = [
    {
        'enabled': False,
        'symbol': 'XAUUSD',
        'candle_table': 'candles_mt5_xauusd_1m',
        'tick_table': 'market_ticks_xauusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
    },
    {
        'enabled': True,
        'symbol': 'JP225.cash',
        'candle_table': 'candles_mt5_jp225_cash_1m',
        'tick_table': 'market_ticks_jp225',
        'tick_size': 1.0,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
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
        print(f"❌ Erreur DB: {e}")
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


def load_all_data(conn, asset):
    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = requested_start - timedelta(hours=2)
    ts_start = int(data_start.timestamp() * 1000)
    
    query_candles = f"SELECT ts, open, high, low, close FROM {asset['candle_table']} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df_candles = pd.read_sql(query_candles, conn)
    df_candles['dt'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)
    
    if df_candles.empty:
        return df_candles, pd.DataFrame()
    
    t_start = data_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = df_candles['dt'].max().strftime("%Y-%m-%d %H:%M:%S")
    
    query_ticks = f"SELECT time, last as price, volume FROM {asset['tick_table']} WHERE symbol = '{asset['symbol']}' AND time >= '{t_start}' AND time <= '{t_end}' ORDER BY time ASC"
    df_ticks = pd.read_sql(query_ticks, conn)
    df_ticks['time'] = pd.to_datetime(df_ticks['time'], utc=True)
    
    return df_candles, df_ticks


def get_session(dt):
    h = dt.hour + dt.minute / 60.0
    if 1 <= h < 8:
        return "TOKYO"
    if 8 <= h < 14.5:
        return "LONDON"
    if 14.5 <= h < 21:
        return "NY"
    return "AUTRE"


def run_backtest():
    conn = get_db_connection()
    enabled_assets = [a for a in ASSETS if a.get('enabled', True)]
    
    if not enabled_assets:
        print("❌ Aucun asset activé.")
        return
    
    print("📊 Chargement des données...")
    t0 = time.time()
    
    assets_data = {}
    for asset in enabled_assets:
        df_candles, df_ticks = load_all_data(conn, asset)
        if not df_candles.empty:
            df_ticks['minute'] = df_ticks['time'].dt.floor('T')
            ticks_by_minute = df_ticks.groupby('minute').apply(lambda g: (g['price'].values, g['volume'].values)).to_dict()
            assets_data[asset['symbol']] = {
                'config': asset,
                'candles': df_candles,
                'ticks_by_minute': ticks_by_minute,
                'vp': IncrementalVolumeProfile(tick_size=asset['tick_size'], va_percent=asset['va_percent']),
                'state': "INSIDE",
                'swing_extreme': 0.0,
                'active_trade': None,
                'breakout_time': None,
                'breakout_price': None,
            }
            print(f"   ✓ {asset['symbol']}: {len(df_candles):,} candles | {len(df_ticks):,} ticks")
    
    conn.close()
    print(f"   ⏱️  Chargé en {time.time() - t0:.2f}s")
    
    if not assets_data:
        print("❌ Aucune donnée chargée.")
        return
    
    all_candles = []
    for symbol, data in assets_data.items():
        df = data['candles'].copy()
        df['symbol'] = symbol
        all_candles.append(df)
    
    df_all_candles = pd.concat(all_candles).sort_values(['dt', 'symbol']).reset_index(drop=True)
    
    sessions_config = {"TOKYO": USE_TOKYO, "LONDON": USE_LONDON, "NY": USE_NY, "AUTRE": False}
    
    current_capital = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    max_dd_amount = 0.0
    max_dd_percent = 0.0  # Max DD en % du peak
    all_trades = []
    
    current_day = None
    day_start_capital = INITIAL_CAPITAL
    day_trades = 0
    day_wins = 0
    day_pnl_r = 0.0
    day_high_water = INITIAL_CAPITAL
    day_max_dd = 0.0
    
    symbols_list = list(assets_data.keys())
    active_sessions = [s for s, v in sessions_config.items() if v]
    directions = []
    if any(a['allow_long'] for a in enabled_assets): directions.append("LONG")
    if any(a['allow_short'] for a in enabled_assets): directions.append("SHORT")
    poc_filter_status = "ON" if FILTER_ENTRY_VS_POC else "OFF"
    trailing_status = "ON" if USE_TRAILING else "OFF"
    tp_mode_display = f"R:R {TARGET_RR}" if TP_MODE == "FIXED_RR" else f"POC (min {MIN_RR}R)"
    
    print(f"\n🚀 Backtest VP Failed Breakout | MULTI-ASSETS: {symbols_list}")
    print(f"💰 Capital: ${INITIAL_CAPITAL:,.2f} | Risque: {RISK_PERCENT*100}% | TP: {tp_mode_display} | Trailing: {trailing_status}")
    print(f"📅 Sessions: {active_sessions} | Directions: {directions} | POC Filter: {poc_filter_status}")
    print("=" * 120)
    print(f"{'DATE':<12} | {'TRADES':>6} | {'WIN':>4} | {'LOSS':>4} | {'WR%':>6} | {'PnL R':>8} | {'CAPITAL':>14} | {'DAY DD%':>8} | {'MAX DD%':>8}")
    print("-" * 120)
    
    t_start = time.time()
    
    for row in df_all_candles.itertuples():
        symbol = row.symbol
        asset_data = assets_data[symbol]
        config = asset_data['config']
        current_minute = row.dt.floor('T')
        row_day = row.dt.date()
        
        if current_day is not None and row_day != current_day:
            day_wr = (day_wins / day_trades * 100) if day_trades > 0 else 0
            day_max_dd_pct = (day_max_dd / day_high_water * 100) if day_high_water > 0 else 0
            print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades - day_wins:>4} | {day_wr:>5.1f}% | {day_pnl_r:>+7.1f}R | ${current_capital:>13,.2f} | {day_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
            day_start_capital = current_capital
            day_trades = 0
            day_wins = 0
            day_pnl_r = 0.0
            day_high_water = current_capital
            day_max_dd = 0.0
        
        current_day = row_day
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
                        tp1_price = active_trade['entry'] - (active_trade['risk'] * MIN_RR)
                        if row.low <= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = MIN_RR * 0.5
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
                        tp1_price = active_trade['entry'] + (active_trade['risk'] * MIN_RR)
                        if row.high >= tp1_price:
                            active_trade['partial_closed'] = True
                            active_trade['sl'] = active_trade['entry']
                            active_trade['partial_pnl_r'] = MIN_RR * 0.5
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
                
                if current_capital > high_water_mark:
                    high_water_mark = current_capital
                if current_capital > day_high_water:
                    day_high_water = current_capital
                
                current_dd = high_water_mark - current_capital
                current_dd_percent = (current_dd / high_water_mark) * 100 if high_water_mark > 0 else 0
                if current_dd > max_dd_amount:
                    max_dd_amount = current_dd
                if current_dd_percent > max_dd_percent:
                    max_dd_percent = current_dd_percent
                
                day_dd = day_high_water - current_capital
                if day_dd > day_max_dd:
                    day_max_dd = day_dd
                
                all_trades.append({
                    'symbol': symbol,
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
                asset_data['active_trade'] = None
            else:
                continue
        
        if row.dt.hour in [23, 0]:
            asset_data['vp'].reset()
            asset_data['state'] = "INSIDE"
            asset_data['swing_extreme'] = 0.0
            continue
        
        curr_sess = get_session(row.dt)
        if not sessions_config.get(curr_sess, False):
            asset_data['state'] = "INSIDE"
            continue
        
        if current_minute in asset_data['ticks_by_minute']:
            prices, volumes = asset_data['ticks_by_minute'][current_minute]
            asset_data['vp'].add_ticks(prices, volumes)
        
        poc, vah, val = asset_data['vp'].get_levels()
        if poc is None:
            continue
        
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
                if config['allow_short']:
                    sl = asset_data['swing_extreme'] + 0.10
                    risk = sl - close
                    if TP_MODE == "POC":
                        tp = poc
                        actual_rr = (close - tp) / risk if risk > 0 else 0
                        
                        # ✅ Filtre anti-aberration
                        if actual_rr > 30:  # R:R trop élevé = risk trop faible
                            continue  # Skip ce trade
        
                    else:
                        tp = close - (risk * TARGET_RR)
                        actual_rr = TARGET_RR
                    poc_ok = (close >= poc) if FILTER_ENTRY_VS_POC else True
                    rr_ok = actual_rr >= MIN_RR
                    if risk > 0 and tp >= val and poc_ok and rr_ok:
                        tp1_price = close - (risk * MIN_RR) if USE_TRAILING and TP_MODE == "POC" else None
                        asset_data['active_trade'] = {
                            'type': 'SHORT', 'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
                            'tp': tp, 'tp1': tp1_price, 'rr': actual_rr, 'session_at_open': curr_sess,
                            'partial_closed': False, 'breakout_time': asset_data['breakout_time'],
                            'breakout_price': asset_data['breakout_price'], 'entry_time': row.dt,
                            'tp1_time': None, 'tp2_time': None, 'exit_time': None,
                            'vah_at_entry': vah, 'val_at_entry': val, 'poc_at_entry': poc
                        }
                asset_data['state'] = "INSIDE"
        
        elif state == "BREAKOUT_DOWN":
            asset_data['swing_extreme'] = min(swing_extreme, low)
            if close > val:
                if config['allow_long']:
                    sl = asset_data['swing_extreme'] - 0.10
                    risk = close - sl
                    if TP_MODE == "POC":
                        tp = poc
                        actual_rr = (tp - close) / risk if risk > 0 else 0
                    else:
                        tp = close + (risk * TARGET_RR)
                        actual_rr = TARGET_RR
                    poc_ok = (close <= poc) if FILTER_ENTRY_VS_POC else True
                    rr_ok = actual_rr >= MIN_RR
                    if risk > 0 and tp <= vah and poc_ok and rr_ok:
                        tp1_price = close + (risk * MIN_RR) if USE_TRAILING and TP_MODE == "POC" else None
                        asset_data['active_trade'] = {
                            'type': 'LONG', 'entry': close, 'sl': sl, 'original_sl': sl, 'risk': risk,
                            'tp': tp, 'tp1': tp1_price, 'rr': actual_rr, 'session_at_open': curr_sess,
                            'partial_closed': False, 'breakout_time': asset_data['breakout_time'],
                            'breakout_price': asset_data['breakout_price'], 'entry_time': row.dt,
                            'tp1_time': None, 'tp2_time': None, 'exit_time': None,
                            'vah_at_entry': vah, 'val_at_entry': val, 'poc_at_entry': poc
                        }
                asset_data['state'] = "INSIDE"
    
    if day_trades > 0:
        day_wr = (day_wins / day_trades * 100) if day_trades > 0 else 0
        day_max_dd_pct = (day_max_dd / day_high_water * 100) if day_high_water > 0 else 0
        print(f"{current_day} | {day_trades:>6} | {day_wins:>4} | {day_trades - day_wins:>4} | {day_wr:>5.1f}% | {day_pnl_r:>+7.1f}R | ${current_capital:>13,.2f} | {day_max_dd_pct:>7.2f}% | {max_dd_percent:>7.2f}%")
    
    elapsed = time.time() - t_start
    
    print("\n" + "=" * 120)
    print("📊 RAPPORT FINAL")
    print("=" * 120)
    
    if not all_trades:
        print("❌ Aucun trade exécuté.")
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
    avg_loss = abs(df_trades[df_trades['result'] == 'LOSS']['pnl_r'].mean()) if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    
    df_trades['is_loss'] = df_trades['result'] == 'LOSS'
    df_trades['streak'] = (df_trades['is_loss'] != df_trades['is_loss'].shift()).cumsum()
    losing_streaks = df_trades[df_trades['is_loss']].groupby('streak').size()
    winning_streaks = df_trades[~df_trades['is_loss']].groupby('streak').size()
    max_losing_streak = losing_streaks.max() if len(losing_streaks) > 0 else 0
    max_winning_streak = winning_streaks.max() if len(winning_streaks) > 0 else 0
    
    recovery_factor = total_pnl / max_dd_amount if max_dd_amount > 0 else float('inf')
    trading_days = df_trades['date'].dt.date.nunique()
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
    
    print(f"\n{'─' * 60}")
    print("📈 PERFORMANCE GLOBALE")
    print(f"{'─' * 60}")
    print(f"  Capital Initial:      ${INITIAL_CAPITAL:>14,.2f}")
    print(f"  Capital Final:        ${current_capital:>14,.2f}")
    print(f"  Profit/Perte:         ${total_pnl:>+14,.2f} ({total_pnl/INITIAL_CAPITAL*100:+.2f}%)")
    print(f"  Gain Total:           {total_pnl_r:>+14.1f} R")
    print(f"  Max Drawdown:         {max_dd_percent:>13.2f}%")
    print(f"  Recovery Factor:      {recovery_factor:>14.2f}")
    
    print(f"\n{'─' * 60}")
    print("🎯 STATISTIQUES DE TRADING")
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
    print("📅 STATISTIQUES TEMPORELLES")
    print(f"{'─' * 60}")
    print(f"  Jours de Trading:     {trading_days:>14}")
    print(f"  Trades/Jour (avg):    {avg_trades_per_day:>14.1f}")
    print(f"  Max Série Gagnante:   {max_winning_streak:>14}")
    print(f"  Max Série Perdante:   {max_losing_streak:>14}")
    print(f"  Temps d'exécution:    {elapsed:>13.2f}s")
    
    # BREAKDOWN PAR ASSET
    print(f"\n{'─' * 60}")
    print("💹 BREAKDOWN PAR ASSET")
    print(f"{'─' * 60}")
    print(f"{'ASSET':<15} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6}")
    print("-" * 80)
    
    for symbol in symbols_list:
        sym_trades = df_trades[df_trades['symbol'] == symbol]
        if len(sym_trades) == 0:
            continue
        s_total = len(sym_trades)
        s_wins = len(sym_trades[sym_trades['result'].isin(['WIN', 'BE'])])
        s_losses = len(sym_trades[sym_trades['result'] == 'LOSS'])
        s_wr = s_wins / s_total * 100
        s_pnl_r = sym_trades['pnl_r'].sum()
        s_gross_profit = sym_trades[sym_trades['pnl'] > 0]['pnl'].sum()
        s_gross_loss = abs(sym_trades[sym_trades['pnl'] < 0]['pnl'].sum())
        s_pf = s_gross_profit / s_gross_loss if s_gross_loss > 0 else float('inf')
        pf_str = f"{s_pf:.2f}" if s_pf != float('inf') else "∞"
        print(f"{symbol:<15} | {s_total:>7} | {s_wins:>5} | {s_losses:>5} | {s_wr:>6.1f}% | {s_pnl_r:>+8.1f}R | {pf_str:>6}")
    
    # BREAKDOWN PAR SESSION
    print(f"\n{'─' * 60}")
    print("🌍 BREAKDOWN PAR SESSION")
    print(f"{'─' * 60}")
    print(f"{'SESSION':<10} | {'TRADES':>7} | {'WIN':>5} | {'LOSS':>5} | {'WR%':>7} | {'PnL R':>9} | {'PF':>6}")
    print("-" * 75)
    
    for session in ["TOKYO", "LONDON", "NY"]:
        sess_trades = df_trades[df_trades['session'] == session]
        if len(sess_trades) == 0:
            continue
        s_total = len(sess_trades)
        s_wins = len(sess_trades[sess_trades['result'].isin(['WIN', 'BE'])])
        s_losses = len(sess_trades[sess_trades['result'] == 'LOSS'])
        s_wr = s_wins / s_total * 100
        s_pnl_r = sess_trades['pnl_r'].sum()
        s_gross_profit = sess_trades[sess_trades['pnl'] > 0]['pnl'].sum()
        s_gross_loss = abs(sess_trades[sess_trades['pnl'] < 0]['pnl'].sum())
        s_pf = s_gross_profit / s_gross_loss if s_gross_loss > 0 else float('inf')
        pf_str = f"{s_pf:.2f}" if s_pf != float('inf') else "∞"
        print(f"{session:<10} | {s_total:>7} | {s_wins:>5} | {s_losses:>5} | {s_wr:>6.1f}% | {s_pnl_r:>+8.1f}R | {pf_str:>6}")
    
    # BREAKDOWN PAR DIRECTION
    print(f"\n{'─' * 60}")
    print("📊 BREAKDOWN PAR DIRECTION")
    print(f"{'─' * 60}")
    
    for trade_type in ["LONG", "SHORT"]:
        type_trades = df_trades[df_trades['type'] == trade_type]
        if len(type_trades) == 0:
            continue
        t_total = len(type_trades)
        t_wins = len(type_trades[type_trades['result'].isin(['WIN', 'BE'])])
        t_losses = len(type_trades[type_trades['result'] == 'LOSS'])
        t_wr = t_wins / t_total * 100
        t_pnl_r = type_trades['pnl_r'].sum()
        t_gross_profit = type_trades[type_trades['pnl'] > 0]['pnl'].sum()
        t_gross_loss = abs(type_trades[type_trades['pnl'] < 0]['pnl'].sum())
        t_pf = t_gross_profit / t_gross_loss if t_gross_loss > 0 else float('inf')
        emoji = "🟢" if trade_type == "LONG" else "🔴"
        pf_str = f"{t_pf:.2f}" if t_pf != float('inf') else "∞"
        print(f"{emoji} {trade_type:<8} | {t_total:>7} trades | {t_wins:>5} W | {t_losses:>5} L | {t_wr:>5.1f}% | {t_pnl_r:>+8.1f}R | PF {pf_str}")
    
    print("=" * 120)
    
    # # 10 DERNIERS TRADES
    # print(f"\n{'─' * 120}")
    # print("🔍 10 DERNIERS TRADES (Validation Manuelle)")
    # print(f"{'─' * 120}")
    
    # last_10 = df_trades.tail(10)
    
    # for idx, trade in last_10.iterrows():
        # def fmt_time(t):
            # return t.strftime('%d/%m/%Y %H:%M') if pd.notna(t) and t is not None else "N/A"
        
        # risk_calc = trade['entry'] - trade['sl'] if trade['type'] == 'LONG' else trade['sl'] - trade['entry']
        
        # if trade['result'] == 'WIN':
            # res_emoji = "✅ WIN "
        # elif trade['result'] == 'BE':
            # res_emoji = "🟡 BE  "
        # else:
            # res_emoji = "❌ LOSS"
        
        # dir_emoji = "🟢 LONG " if trade['type'] == 'LONG' else "🔴 SHORT"
        
        # print(f"\n  ╔{'═' * 100}╗")
        # print(f"  ║ Trade #{idx + 1:03d} | {trade['symbol']:<12} | {dir_emoji} | {trade['session']:<6} | {res_emoji} | PnL: {trade['pnl_r']:+.2f}R")
        # print(f"  ╠{'═' * 100}╣")
        
        # vah = trade.get('vah_at_entry', 'N/A')
        # val = trade.get('val_at_entry', 'N/A')
        # poc = trade.get('poc_at_entry', 'N/A')
        # vah_str = f"{vah:.2f}" if isinstance(vah, (int, float)) else vah
        # val_str = f"{val:.2f}" if isinstance(val, (int, float)) else val
        # poc_str = f"{poc:.2f}" if isinstance(poc, (int, float)) else poc
        
        # print(f"  ║ 📊 VP LEVELS:  VAH: {vah_str}  |  POC: {poc_str}  |  VAL: {val_str}")
        # print(f"  ╠{'─' * 100}╣")
        # print(f"  ║ ⏱️  TIMELINE:")
        # print(f"  ║    1️⃣  BREAKOUT:      {fmt_time(trade.get('breakout_time')):<20} | Prix: {trade.get('breakout_price', 'N/A')}")
        # print(f"  ║    2️⃣  ENTRY:         {fmt_time(trade.get('entry_time')):<20} | Prix: {trade['entry']:.2f}")
        
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
        
        # exit_price = trade.get('exit_price')
        # exit_price_str = f"{exit_price:.2f}" if exit_price else "N/A"
        # print(f"  ║    🏁 EXIT:          {fmt_time(trade.get('exit_time')):<20} | Prix: {exit_price_str}")
        # print(f"  ╠{'─' * 100}╣")
        # print(f"  ║ 🎯 TRADE LEVELS:")
        # print(f"  ║    Entry:  {trade['entry']:.2f}")
        # print(f"  ║    SL:     {trade['sl']:.2f}  (Risk: {risk_calc:.2f})")
        # if USE_TRAILING and TP_MODE == "POC" and trade.get('tp1'):
            # print(f"  ║    TP1:    {trade['tp1']:.2f}  (à {MIN_RR}R = 50% position)")
            # print(f"  ║    TP2:    {trade['tp']:.2f}  (POC = 50% restant)")
        # else:
            # print(f"  ║    TP:     {trade['tp']:.2f}")
        # print(f"  ║    R:R:    {trade['rr']:.2f}")
        # print(f"  ╠{'─' * 100}╣")
        # print(f"  ║ 💰 Capital après: ${trade['capital_after']:,.2f}")
        # print(f"  ╚{'═' * 100}╝")
    
    # print(f"\n{'─' * 120}")
    # print("✅ Fin du rapport - Copiez les timestamps ci-dessus pour vérifier sur TradingView")
    # print("=" * 120)
    
    
    # # TRADES > 10R
    # print(f"\n{'─' * 120}")
    # print("🚀 TRADES AYANT FAIT PLUS DE 10R")
    # print(f"{'─' * 120}")
    
    # trades_10r_plus = df_trades[df_trades['pnl_r'] > 5.0].sort_values('pnl_r', ascending=False)
    
    # if len(trades_10r_plus) == 0:
        # print("❌ Aucun trade > 10R")
    # else:
        # print(f"✅ {len(trades_10r_plus)} trade(s) > 10R trouvé(s)\n")
        
        # for idx, trade in trades_10r_plus.iterrows():
            # def fmt_time(t):
                # return t.strftime('%d/%m/%Y %H:%M') if pd.notna(t) and t is not None else "N/A"
            
            # risk_calc = trade['entry'] - trade['sl'] if trade['type'] == 'LONG' else trade['sl'] - trade['entry']
            
            # if trade['result'] == 'WIN':
                # res_emoji = "✅ WIN "
            # elif trade['result'] == 'BE':
                # res_emoji = "🟡 BE  "
            # else:
                # res_emoji = "❌ LOSS"
            
            # dir_emoji = "🟢 LONG " if trade['type'] == 'LONG' else "🔴 SHORT"
            
            # print(f"\n  ╔{'═' * 100}╗")
            # print(f"  ║ Trade #{idx + 1:03d} | {trade['symbol']:<12} | {dir_emoji} | {trade['session']:<6} | {res_emoji} | PnL: {trade['pnl_r']:+.2f}R ⭐")
            # print(f"  ╠{'═' * 100}╣")
            
            # vah = trade.get('vah_at_entry', 'N/A')
            # val = trade.get('val_at_entry', 'N/A')
            # poc = trade.get('poc_at_entry', 'N/A')
            # vah_str = f"{vah:.2f}" if isinstance(vah, (int, float)) else vah
            # val_str = f"{val:.2f}" if isinstance(val, (int, float)) else val
            # poc_str = f"{poc:.2f}" if isinstance(poc, (int, float)) else poc
            
            # print(f"  ║ 📊 VP LEVELS:  VAH: {vah_str}  |  POC: {poc_str}  |  VAL: {val_str}")
            # print(f"  ╠{'─' * 100}╣")
            # print(f"  ║ ⏱️  TIMELINE:")
            # print(f"  ║    1️⃣  BREAKOUT:      {fmt_time(trade.get('breakout_time')):<20} | Prix: {trade.get('breakout_price', 'N/A')}")
            # print(f"  ║    2️⃣  ENTRY:         {fmt_time(trade.get('entry_time')):<20} | Prix: {trade['entry']:.2f}")
            
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
            
            # exit_price = trade.get('exit_price')
            # exit_price_str = f"{exit_price:.2f}" if exit_price else "N/A"
            # print(f"  ║    🏁 EXIT:          {fmt_time(trade.get('exit_time')):<20} | Prix: {exit_price_str}")
            # print(f"  ╠{'─' * 100}╣")
            # print(f"  ║ 🎯 TRADE LEVELS:")
            # print(f"  ║    Entry:  {trade['entry']:.2f}")
            # print(f"  ║    SL:     {trade['sl']:.2f}  (Risk: {risk_calc:.2f})")
            # if USE_TRAILING and TP_MODE == "POC" and trade.get('tp1'):
                # print(f"  ║    TP1:    {trade['tp1']:.2f}  (à {MIN_RR}R = 50% position)")
                # print(f"  ║    TP2:    {trade['tp']:.2f}  (POC = 50% restant)")
            # else:
                # print(f"  ║    TP:     {trade['tp']:.2f}")
            # print(f"  ║    R:R:    {trade['rr']:.2f}")
            # print(f"  ╠{'─' * 100}╣")
            # print(f"  ║ 💰 Capital après: ${trade['capital_after']:,.2f}")
            # print(f"  ╚{'═' * 100}╝")


if __name__ == "__main__":
    run_backtest()