#!/usr/bin/env python3
# postgres_xrp_deep_analysis.py

import os
import re
import csv
import sys
import argparse
import bisect 
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from colorama import init, Fore, Style

init(autoreset=True)
UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d %H:%M"

# ---------- CONFIGURATION ----------
TARGET_PAIR = "XRPUSD.c"    # La cible
SCAN_TF = "15m"
MAX_WAIT_CANDLES = 32       # 8 heures de validité pour l'ordre limite
RR_TARGET = Decimal("2.0")  
RISK_PER_TRADE = Decimal("0.01") # 1% du capital courant
INITIAL_CAPITAL = Decimal("50000.00")
FEES_PCT = Decimal("0.05")

EMA_PERIOD = 200            
FIB_LEVEL = 0.618           
SWING_LEN = 5               

# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime(DATE_FORMAT)

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    return create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")

def price_scale(pair: str) -> int:
    return 5 # XRP a souvent besoin de précision, ou 4

# -----------------------------------------------------------
# MOTEUR (Strictement identique à la version validée)
# -----------------------------------------------------------

def add_indicators(df: pd.DataFrame):
    df['ema200'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    l = SWING_LEN
    highs = df['high'].values
    lows = df['low'].values
    df['is_swing_high'] = False
    df['is_swing_low'] = False
    
    for i in range(l, len(df) - l):
        if highs[i] == max(highs[i-l:i+l+1]):
            df.at[i, 'is_swing_high'] = True
        if lows[i] == min(lows[i-l:i+l+1]):
            df.at[i, 'is_swing_low'] = True
    return df

def detect_setup(rates, i):
    vision_limit_idx = i - SWING_LEN
    if vision_limit_idx < 200: return None 

    current_close = rates[i]['close']
    ema_value = rates[i]['ema200']
    
    if pd.isna(ema_value): return None

    is_uptrend = current_close > ema_value
    is_downtrend = current_close < ema_value
    
    # LONG
    if is_uptrend:
        last_sh_price = -1.0
        last_sh_idx = -1
        for k in range(vision_limit_idx, vision_limit_idx - 50, -1):
            if rates[k]['is_swing_high']:
                last_sh_price = rates[k]['high']
                last_sh_idx = k
                break
        
        if last_sh_idx == -1: return None
        
        last_sl_price = 999999.0
        last_sl_idx = -1
        for k in range(last_sh_idx - 1, last_sh_idx - 100, -1):
            if rates[k]['is_swing_low']:
                last_sl_price = rates[k]['low']
                last_sl_idx = k
                break
                
        if last_sl_idx != -1:
            range_size = last_sh_price - last_sl_price
            fib_entry = last_sh_price - (range_size * FIB_LEVEL)
            sl_price = last_sl_price
            
            if fib_entry < ema_value: return None
            if current_close > fib_entry:
                tp_price = fib_entry + (abs(fib_entry - sl_price) * float(RR_TARGET))
                return {"side": "LONG", "entry": fib_entry, "sl": sl_price, "tp": tp_price}

    # SHORT
    elif is_downtrend:
        last_sl_price = 999999.0
        last_sl_idx = -1
        for k in range(vision_limit_idx, vision_limit_idx - 50, -1):
            if rates[k]['is_swing_low']:
                last_sl_price = rates[k]['low']
                last_sl_idx = k
                break
        
        if last_sl_idx == -1: return None
        
        last_sh_price = -1.0
        last_sh_idx = -1
        for k in range(last_sl_idx - 1, last_sl_idx - 100, -1):
            if rates[k]['is_swing_high']:
                last_sh_price = rates[k]['high']
                last_sh_idx = k
                break
                
        if last_sh_idx != -1:
            range_size = last_sh_price - last_sl_price
            fib_entry = last_sl_price + (range_size * FIB_LEVEL)
            sl_price = last_sh_price
            
            if fib_entry > ema_value: return None
            if current_close < fib_entry:
                tp_price = fib_entry - (abs(sl_price - fib_entry) * float(RR_TARGET))
                return {"side": "SHORT", "entry": fib_entry, "sl": sl_price, "tp": tp_price}

    return None

def run_simulation(ltf_data, start_idx, setup, expiration_ts):
    is_filled = False
    entry = float(setup['entry'])
    sl = float(setup['sl'])
    tp = float(setup['tp'])
    side = setup['side']
    limit = min(start_idx + 10000, len(ltf_data))
    
    for i in range(start_idx, limit):
        row = ltf_data[i]
        ts, h, l = row['ts'], float(row['high']), float(row['low'])
        
        if ts > expiration_ts: return "EXPIRED", ts, 0.0
        
        if not is_filled:
            if side == "LONG":
                if l <= entry: 
                    is_filled = True
                    if l <= sl: return "LOSS", ts, sl
            elif side == "SHORT":
                if h >= entry:
                    is_filled = True
                    if h >= sl: return "LOSS", ts, sl
        else: 
            if side == "LONG":
                if l <= sl: return "LOSS", ts, sl
                if h >= tp: return "WIN", ts, tp
            elif side == "SHORT":
                if h >= sl: return "LOSS", ts, sl
                if l <= tp: return "WIN", ts, tp

    return "EXPIRED", ltf_data[limit-1]['ts'] if ltf_data else expiration_ts, 0.0

# -----------------------------------------------------------
# EXECUTION & REPORTING
# -----------------------------------------------------------

def main():
    engine = get_pg_engine()
    print(Fore.CYAN + f"--- ANALYSE DETAILLÉE : {TARGET_PAIR} ---")
    print(f"Stratégie: EMA {EMA_PERIOD} + Fib {FIB_LEVEL}")
    print("Chargement des données...")

    # Fetch 15m
    table = f"candles_mt5_{sanitize_name(TARGET_PAIR)}_{sanitize_name(SCAN_TF)}"
    df = pd.read_sql(f"SELECT ts as time, open, high, low, close FROM {table} ORDER BY ts ASC", engine)
    if df.empty: return
    df = add_indicators(df)
    htf_rates = df.to_dict('records')

    # Fetch 1m
    table_ltf = f"candles_mt5_{sanitize_name(TARGET_PAIR)}_1m"
    df_ltf = pd.read_sql(f"SELECT ts, high, low FROM {table_ltf} ORDER BY ts ASC", engine)
    ltf_data = df_ltf.to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data]

    print(f"Données chargées. {len(htf_rates)} bougies 15m. Début simulation...")
    
    trades = []
    scan_ms = 15 * 60 * 1000
    skip_until = 0
    balance = INITIAL_CAPITAL
    
    scale = price_scale(TARGET_PAIR)
    
    # Headers Console
    print("\n" + "="*100)
    print(f"{'DATE SETUP':<18} | {'SIDE':<5} | {'ENTRY':<10} | {'SL':<10} | {'TP':<10} | {'RESULT':<6} | {'PNL($)':<10} | {'BALANCE':<10}")
    print("-" * 100)

    for i in range(250, len(htf_rates)):
        row = htf_rates[i]
        if row['time'] < skip_until: continue
        
        setup = detect_setup(htf_rates, i)
        
        if setup:
            sim_start = row['time'] + scan_ms
            l_idx = bisect.bisect_left(ltf_ts, sim_start)
            if l_idx >= len(ltf_data): continue
            
            expire = sim_start + (MAX_WAIT_CANDLES * scan_ms)
            
            res, exit_ts, exit_price = run_simulation(ltf_data, l_idx, setup, expire)
            
            if res in ["WIN", "LOSS"]:
                # Gestion Capital
                risk_amt = balance * RISK_PER_TRADE
                pnl_r = (RR_TARGET if res == "WIN" else Decimal("-1.0")) - FEES_PCT
                pnl_usd = risk_amt * pnl_r
                balance += pnl_usd
                
                # Console Print
                col = Fore.GREEN if res == "WIN" else Fore.RED
                date_str = format_ts(row['time'])
                
                print(f"{date_str:<18} | {setup['side']:<5} | {qround(setup['entry'], scale):<10} | {qround(setup['sl'], scale):<10} | {qround(setup['tp'], scale):<10} | {col}{res:<6}{Fore.RESET} | {pnl_usd:>9.2f} | {balance:>10.2f}")
                
                trades.append({
                    "date": date_str,
                    "side": setup['side'],
                    "entry": setup['entry'],
                    "sl": setup['sl'],
                    "tp": setup['tp'],
                    "exit_price": exit_price,
                    "result": res,
                    "pnl_r": pnl_r,
                    "pnl_usd": pnl_usd,
                    "balance": balance
                })
                skip_until = exit_ts

    # Export CSV
    if trades:
        csv_file = "xrp_backtest_details.csv"
        keys = trades[0].keys()
        with open(csv_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(trades)
            
        print("\n" + "="*100)
        print(f"ANALYSE TERMINEE.")
        print(f"Total Trades: {len(trades)}")
        print(f"Solde Final:  {balance:,.2f}$")
        print(f"Performance:  {((balance - INITIAL_CAPITAL)/INITIAL_CAPITAL)*100:.2f}%")
        print(f"Détails exportés dans : {csv_file}")
    else:
        print("Aucun trade trouvé sur la période.")

if __name__ == "__main__":
    main()