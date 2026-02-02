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

# FILTRES DE SESSION
USE_TOKYO = True
USE_LONDON = False
USE_NY = False

# DATE DE DÉBUT
START_DATE_STR = "2025-07-01 00:00:00"

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
    req_start = datetime.strptime(START_DATE_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    data_start = req_start - timedelta(hours=2) 
    ts_start = int(data_start.timestamp() * 1000)
    query = f"SELECT ts, open, high, low, close FROM {CANDLE_TABLE} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df = pd.read_sql(query, conn)
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df

def get_ticks_for_window(conn, dt_session_start, dt_current_candle):
    t_start = dt_session_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = dt_current_candle.strftime("%Y-%m-%d %H:%M:%S")
    query = f"SELECT last as price, volume FROM {TICK_TABLE} WHERE symbol = '{SYMBOL}' AND time >= '{t_start}' AND time <= '{t_end}'"
    return pd.read_sql(query, conn)

# --- SVP CALCUL (Gelé) ---
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
    if 0 <= h < 6: return "TOKYO"
    if 8 <= h < 14.5: return "LONDON"
    if 14.5 <= h < 21: return "NY"
    return "AUTRE"

# --- MOTEUR DE STRATÉGIE ---
def run_backtest():
    conn = get_db_connection()
    df_candles = get_candles_stream(conn)
    if df_candles.empty: return

    session_start_dt = df_candles.iloc[0]['dt']
    active_trade = None 
    last_row = None # Stockage bougie i-1
    last_vah, last_val = None, None

    sessions_config = {"TOKYO": USE_TOKYO, "LONDON": USE_LONDON, "NY": USE_NY, "AUTRE": False}
    session_stats = {s: {"wins": 0, "losses": 0, "total_r": 0.0, "peak_r": 0.0, "max_dd": 0.0} for s in sessions_config.keys()}
    
    print(f"🔄 Backtest V4 (Trap & Wickout RR1) | Start: {START_DATE_STR}")
    print("-" * 115)

    for row in df_candles.itertuples():
        # 1. GESTION DU TRADE ACTIF
        if active_trade:
            res = None
            if active_trade['type'] == 'SHORT':
                if row.high >= active_trade['sl']: res = "❌ LOSS"
                elif row.low <= active_trade['tp']: res = "✅ WIN"
            elif active_trade['type'] == 'LONG':
                if row.low <= active_trade['sl']: res = "❌ LOSS"
                elif row.high >= active_trade['tp']: res = "✅ WIN"
            
            if res:
                s_name = active_trade['session_at_open']
                if "WIN" in res:
                    session_stats[s_name]["wins"] += 1
                    session_stats[s_name]["total_r"] += 1.0
                else:
                    session_stats[s_name]["losses"] += 1
                    session_stats[s_name]["total_r"] -= 1.0
                
                curr_r = session_stats[s_name]["total_r"]
                session_stats[s_name]["peak_r"] = max(session_stats[s_name]["peak_r"], curr_r)
                dd = session_stats[s_name]["peak_r"] - curr_r
                session_stats[s_name]["max_dd"] = max(session_stats[s_name]["max_dd"], dd)

                print(f"   >>> RÉSULTAT TRADE : {res} (Sortie: {row.close} | R Cumul: {curr_r:.2f})")
                active_trade = None 
            continue

        # 2. ZONE DE RESET (23h-01h)
        if row.dt.hour in [23, 0]:
            if row.dt.hour == 23: session_start_dt = row.dt.replace(minute=0, second=0, microsecond=0)
            else: session_start_dt = (row.dt - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
            last_row = None
            continue

        # 3. FILTRE SESSION
        curr_sess = get_session(row.dt)
        if not sessions_config.get(curr_sess, False):
            last_row = None
            continue

        # 4. CALCUL SVP
        df_ticks = get_ticks_for_window(conn, session_start_dt, row.dt)
        poc, vah, val = calculate_svp(df_ticks)
        
        # On a besoin de la bougie i-1 ET des niveaux SVP de la bougie i-1 pour valider le break
        if not poc or last_row is None or last_vah is None:
            last_row = row
            last_vah, last_val = vah, val
            continue

        close, high, low, open_p = row.close, row.high, row.low, row.open
        action_log = ""

        # 5. LOGIQUE SIGNAUX
        
        # --- LOGIQUE SHORT (VAH) ---
        is_short = False
        # A: Clôture i-1 > VAH et Clôture i < VAH
        if last_row.close > last_vah and close < vah:
            is_short = True
        # B: Mèche i-1 > VAH (mais Close < VAH) et Bougie i rouge
        elif last_row.high > last_vah and last_row.close < last_vah and close < open_p:
            is_short = True

        if is_short:
            sl = max(last_row.high, high) + 0.05
            risk = sl - close
            if risk > 0:
                tp = close - risk
                active_trade = {'type': 'SHORT', 'entry': close, 'sl': sl, 'tp': tp, 'session_at_open': curr_sess}
                action_log = f"🔴 SHORT SIGNAL ! Entry: {close} | SL: {sl:.2f}"

        # --- LOGIQUE LONG (VAL) ---
        is_long = False
        if not active_trade:
            # A: Clôture i-1 < VAL et Clôture i > VAL
            if last_row.close < last_val and close > val:
                is_long = True
            # B: Mèche i-1 < VAL (mais Close > VAL) et Bougie i verte
            elif last_row.low < last_val and last_row.close > last_val and close > open_p:
                is_long = True

            if is_long:
                sl = min(last_row.low, low) - 0.05
                risk = close - sl
                if risk > 0:
                    tp = close + risk
                    active_trade = {'type': 'LONG', 'entry': close, 'sl': sl, 'tp': tp, 'session_at_open': curr_sess}
                    action_log = f"🟢 LONG SIGNAL ! Entry: {close} | SL: {sl:.2f}"

        if action_log: 
            print(f"{row.dt.strftime('%d/%m %H:%M')} | {close:<8.2f} | {action_log}")

        last_row = row
        last_vah, last_val = vah, val

    conn.close()

    # --- TABLEAU FINAL ---
    print("\n" + "="*110)
    print(f"{'SESSION':<18} | {'TRADES':<8} | {'WINRATE':<10} | {'TOTAL R':<12} | {'MAX DD (R)':<12}")
    print("-" * 110)
    for s in ["TOKYO", "LONDON", "NY"]:
        d = session_stats[s]; total = d['wins'] + d['losses']
        wr = f"{(d['wins']/total*100):.2f}%" if total > 0 else "N/A"
        print(f"{s + ' (ON)' if sessions_config[s] else s + ' (OFF)':<18} | {total:<8} | {wr:<10} | {d['total_r']:<12.2f} | {d['max_dd']:<12.2f}")
    print("="*110)

if __name__ == "__main__":
    run_backtest()