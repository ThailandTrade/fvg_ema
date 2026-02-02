import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import sys
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

# --- CONFIGURATION ---
SYMBOL = "XAUUSD"
CANDLE_TABLE = "candles_mt5_xauusd_1m"
TICK_TABLE = "market_ticks"
TICK_SIZE = 0.01
VA_PERCENT = 0.70

# PARAMÈTRE MODIFIÉ
RISK_REWARD_RATIO = 2  # Au lieu de 1:1 !

# FILTRES DE SESSION
USE_TOKYO = True
USE_LONDON = False
USE_NY = False

# DATE DE DÉBUT
START_DATE_STR = "2025-08-01 00:00:00"

# --- DB & DATA ---
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

def get_candles_stream(conn):
    requested_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = requested_start - timedelta(hours=2)
    ts_start = int(data_start.timestamp() * 1000)
    
    query = f"""
        SELECT ts, open, high, low, close
        FROM {CANDLE_TABLE}
        WHERE ts >= {ts_start}
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, conn)
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df

def get_ticks_for_window(conn, dt_session_start, dt_current_candle):
    t_start = dt_session_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = dt_current_candle.strftime("%Y-%m-%d %H:%M:%S")
    
    query = f"""
        SELECT last as price, volume FROM {TICK_TABLE}
        WHERE symbol = '{SYMBOL}' AND time >= '{t_start}' AND time <= '{t_end}'
    """
    return pd.read_sql(query, conn)

# --- SVP CALCUL (Original) ---
def calculate_svp(df_ticks):
    if df_ticks.empty: return None, None, None
    
    df_ticks['price_bin'] = (df_ticks['price'] / TICK_SIZE).round() * TICK_SIZE
    profile = df_ticks.groupby('price_bin')['volume'].sum().reset_index().sort_values('price_bin').reset_index(drop=True)
   
    total_vol = profile['volume'].sum()
    target = total_vol * VA_PERCENT
    poc_idx = profile['volume'].idxmax()
    poc_price = profile.iloc[poc_idx]['price_bin']
   
    curr, up, down = profile.iloc[poc_idx]['volume'], poc_idx + 1, poc_idx - 1
    while curr < target:
        v_up = profile.iloc[up]['volume'] if up < len(profile) else 0
        v_down = profile.iloc[down]['volume'] if down >= 0 else 0
        if v_up == 0 and v_down == 0: break
        if v_up > v_down: curr += v_up; up += 1
        else: curr += v_down; down -= 1
           
    return poc_price, profile.iloc[min(up-1, len(profile)-1)]['price_bin'], profile.iloc[max(down+1, 0)]['price_bin']

# --- HELPER SESSION ---
def get_session(dt):
    h = dt.hour + dt.minute/60.0
    if 0 <= h < 8: return "TOKYO"
    if 8 <= h < 14.5: return "LONDON"
    if 14.5 <= h < 21: return "NY"
    return "AUTRE"

# --- MOTEUR DE STRATÉGIE (Original avec RR modifié) ---
def run_backtest():
    conn = get_db_connection()
    df_candles = get_candles_stream(conn)
   
    if df_candles.empty:
        print("❌ Aucune donnée.")
        return
    
    session_start_dt = df_candles.iloc[0]['dt']
    state = "INSIDE"
    swing_extreme = 0.0
    active_trade = None
    current_day = None
    
    sessions_config = {"TOKYO": USE_TOKYO, "LONDON": USE_LONDON, "NY": USE_NY, "AUTRE": False}
    session_stats = {s: {"wins": 0, "losses": 0, "total": 0} for s in sessions_config.keys()}
    
    stats_wins = 0
    stats_losses = 0
    stats_total = 0
    cumulative_r = 0.0
    max_r = 0.0  # Peak R
    max_dd = 0.0  # Drawdown max
    
    print("=" * 120)
    print(f"SVP TOKYO - SCRIPT ORIGINAL | RR: 1:{RISK_REWARD_RATIO}")
    print("=" * 120)
    
    for row in df_candles.itertuples():
        # Changement de jour
        if current_day is None or row.dt.date() != current_day:
            if current_day is not None:
                print("\n" + "-" * 120)
            print(f"{row.dt.strftime('%A %d %B %Y').upper()}")
            print("-" * 120)
            current_day = row.dt.date()
        
        # 1. GESTION DU TRADE ACTIF
        if active_trade:
            res = None
            exit_price = None
            
            if active_trade['type'] == 'SHORT':
                if row.high >= active_trade['sl']: 
                    res = "LOSS"
                    exit_price = active_trade['sl']
                elif row.low <= active_trade['tp']: 
                    res = "WIN"
                    exit_price = active_trade['tp']
            elif active_trade['type'] == 'LONG':
                if row.low <= active_trade['sl']: 
                    res = "LOSS"
                    exit_price = active_trade['sl']
                elif row.high >= active_trade['tp']: 
                    res = "WIN"
                    exit_price = active_trade['tp']
           
            if res:
                # Calcul en R
                if active_trade['type'] == 'SHORT':
                    pl_price = active_trade['entry'] - exit_price
                else:
                    pl_price = exit_price - active_trade['entry']
                
                risk = active_trade['risk']
                pl_r = pl_price / risk
                
                stats_total += 1
                cumulative_r += pl_r
                
                # Calcul Max DD depuis le début
                if cumulative_r > max_r:
                    max_r = cumulative_r  # Nouveau peak
                
                current_dd = max_r - cumulative_r  # DD actuel
                
                if current_dd > max_dd:
                    max_dd = current_dd  # On garde le pire DD de l'histoire
                
                if res == "WIN":
                    stats_wins += 1
                    emoji = "WIN "
                else:
                    stats_losses += 1
                    emoji = "LOSS"
               
                s_name = active_trade['session_at_open']
                session_stats[s_name]["total"] += 1
                if res == "WIN": 
                    session_stats[s_name]["wins"] += 1
                else: 
                    session_stats[s_name]["losses"] += 1
                
                wr = (stats_wins / stats_total * 100)
                print(f"{emoji} | Exit: {exit_price:.2f} | P/L: {pl_r:+.2f}R | Cumul: {cumulative_r:+.2f}R | Max DD: {max_dd:.2f}R | WR: {wr:.1f}%")
                
                active_trade = None
            else:
                continue
        
        # 2. ZONE DE RESET (23h00 - 01h00 UTC)
        if row.dt.hour in [23, 0]:
            if row.dt.hour == 23:
                session_start_dt = row.dt.replace(minute=0, second=0, microsecond=0)
            else:
                session_start_dt = (row.dt - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
            state = "INSIDE"
            swing_extreme = 0.0
            continue
        
        # 3. FILTRE DE SESSION ÉCONOMIQUE
        curr_sess = get_session(row.dt)
        if not sessions_config.get(curr_sess, False):
            state = "INSIDE"
            continue
        
        # 4. CALCUL SVP
        df_ticks = get_ticks_for_window(conn, session_start_dt, row.dt)
        poc, vah, val = calculate_svp(df_ticks)
        if not poc: continue
        
        close = row.close
        high = row.high
        low = row.low
        
        # 5. LOGIQUE STRATÉGIE (Original)
        if state == "INSIDE":
            if close > vah:
                state = "BREAKOUT_UP"
                swing_extreme = high
                print(f"UP  {row.dt.strftime('%H:%M')} | Breakout VAH ({vah:.2f})")
            elif close < val:
                state = "BREAKOUT_DOWN"
                swing_extreme = low
                print(f"DOWN  {row.dt.strftime('%H:%M')} | Breakout VAL ({val:.2f})")
        
        elif state == "BREAKOUT_UP":
            swing_extreme = max(swing_extreme, high)
            
            if close < vah:
                print(f"   <  {row.dt.strftime('%H:%M')} | Pullback sous VAH")
                
                # MODIFIÉ: RR amélioré
                sl = swing_extreme + 0.10
                risk = sl - close
                tp = close - (risk * RISK_REWARD_RATIO)  # TP basé sur RR
                
                if risk > 0 and tp >= val:  # TP doit au moins atteindre VAL
                    active_trade = {
                        'type': 'SHORT', 
                        'entry': close, 
                        'sl': sl, 
                        'tp': tp, 
                        'risk': risk,
                        'session_at_open': curr_sess
                    }
                    print(f"SHORT @ {close:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | RR: 1:{RISK_REWARD_RATIO}")
                    state = "INSIDE"
                else:
                    state = "INSIDE"
        
        elif state == "BREAKOUT_DOWN":
            swing_extreme = min(swing_extreme, low)
            
            if close > val:
                print(f"   >  {row.dt.strftime('%H:%M')} | Pullback au-dessus VAL")
                
                # MODIFIÉ: RR amélioré
                sl = swing_extreme - 0.10
                risk = close - sl
                tp = close + (risk * RISK_REWARD_RATIO)  # TP basé sur RR
                
                if risk > 0 and tp <= vah and close <= poc:
                    active_trade = {
                        'type': 'LONG', 
                        'entry': close, 
                        'sl': sl, 
                        'tp': tp, 
                        'risk': risk,
                        'session_at_open': curr_sess
                    }
                    print(f"LONG @ {close:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | RR: 1:{RISK_REWARD_RATIO}")
                    state = "INSIDE"
                else:
                    state = "INSIDE"
    
    conn.close()
    
    # --- TABLEAU FINAL ---
    print("\n" + "=" * 120)
    print("RESULTATS FINAUX")
    print("=" * 120)
    
    if stats_total > 0:
        wr = (stats_wins / stats_total * 100)
        avg_r = cumulative_r / stats_total
        
        print(f"Total Trades: {stats_total}")
        print(f"Wins: {stats_wins} | Losses: {stats_losses} | Win Rate: {wr:.2f}%")
        print(f"Cumulative R: {cumulative_r:+.2f}R")
        print(f"Max Drawdown: {max_dd:.2f}R")
        print(f"Expectancy: {avg_r:+.2f}R par trade")
        
        print("\nStats par Session:")
        for s in ["TOKYO", "LONDON", "NY"]:
            d = session_stats[s]
            if d['total'] > 0:
                wr_s = (d['wins'] / d['total'] * 100)
                print(f"  {s}: {d['total']} trades | WR: {wr_s:.1f}%")
    else:
        print("Aucun trade execute")
    
    print("=" * 120)

if __name__ == "__main__":
    run_backtest()