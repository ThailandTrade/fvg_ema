#!/usr/bin/env python3
# postgres_ema_fib_golden.py

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

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIGURATION ----------
SCAN_TF = "15m"
MAX_WAIT_CANDLES = 32       # On laisse l'ordre plus longtemps (8h), les gros pullbacks prennent du temps
RR_TARGET = Decimal("2.0")  # On veut du 2R minimum pour payer les pertes
RISK_PER_TRADE = Decimal("0.01")
FEES_PCT = Decimal("0.05")

EMA_PERIOD = 50           # Le Juge de Paix
FIB_LEVEL = 0.618           # Le Golden Ratio
SWING_LEN = 5               # Confirmation Lag

# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def price_scale(base: str, quote: str) -> int:
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def parse_date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, DATE_FORMAT).replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    return create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")

def parse_pairs(path: str):
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r.get("pair").strip() for r in csv.DictReader(f) if r.get("pair")]

# -----------------------------------------------------------
# INDICATEURS
# -----------------------------------------------------------

def add_indicators(df: pd.DataFrame):
    # 1. EMA 200
    df['ema200'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # 2. Swings (Strict)
    # On pré-calcule, mais l'usage sera restreint par l'index
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

# -----------------------------------------------------------
# LOGIQUE STRATEGIQUE (EMA + FIB)
# -----------------------------------------------------------

def detect_ema_fib_setup(rates, i):
    # 1. LIMITES DE VISION (Pas de triche)
    vision_limit_idx = i - SWING_LEN
    if vision_limit_idx < 200: return None # Besoin d'historique pour EMA

    current_close = rates[i]['close']
    ema_value = rates[i]['ema200']
    
    # Si EMA pas calculée
    if pd.isna(ema_value): return None

    # 2. DEFINIR LA TENDANCE DE FOND
    is_uptrend = current_close > ema_value
    is_downtrend = current_close < ema_value
    
    setup = None

    # --- SETUP LONG (TENDANCE HAUSSIERE) ---
    if is_uptrend:
        # On cherche le dernier Swing High CONFIRMÉ (Le sommet de l'impulsion)
        last_sh_price = -1.0
        last_sh_idx = -1
        
        # On scanne en arrière depuis la limite de vision
        for k in range(vision_limit_idx, vision_limit_idx - 50, -1):
            if rates[k]['is_swing_high']:
                last_sh_price = rates[k]['high']
                last_sh_idx = k
                break
        
        if last_sh_idx == -1: return None
        
        # On cherche le Swing Low CONFIRMÉ qui précède ce High (Le début de l'impulsion)
        last_sl_price = 999999.0
        last_sl_idx = -1
        
        for k in range(last_sh_idx - 1, last_sh_idx - 100, -1):
            if rates[k]['is_swing_low']:
                last_sl_price = rates[k]['low']
                last_sl_idx = k
                break
                
        if last_sl_idx != -1:
            # On a une jambe complète (Low -> High)
            # VERIFICATION : Est-ce que cette jambe est "saine" ? (Le Low doit être proche ou au-dessus de l'EMA)
            # Optionnel, mais ici on veut juste suivre la tendance.
            
            # CALCUL FIB 61.8
            range_size = last_sh_price - last_sl_price
            fib_entry = last_sh_price - (range_size * FIB_LEVEL)
            sl_price = last_sl_price # SL sous le début du mouvement
            
            # FILTRE EMA : On n'achète pas si le Fib 61.8 est SOUS l'EMA 200
            # (On ne veut pas acheter un retournement de tendance, juste un pullback)
            if fib_entry < ema_value: 
                return None
                
            # TRIGGER : 
            # 1. Le prix actuel doit être au-dessus de l'entrée (en attente de pullback)
            # 2. Ou juste en train de toucher.
            if current_close > fib_entry:
                tp_price = fib_entry + (abs(fib_entry - sl_price) * float(RR_TARGET))
                return {
                    "side": "LONG",
                    "entry": fib_entry,
                    "sl": sl_price,
                    "tp": tp_price
                }

    # --- SETUP SHORT (TENDANCE BAISSIERE) ---
    elif is_downtrend:
        # Dernier Swing Low Confirmé
        last_sl_price = 999999.0
        last_sl_idx = -1
        
        for k in range(vision_limit_idx, vision_limit_idx - 50, -1):
            if rates[k]['is_swing_low']:
                last_sl_price = rates[k]['low']
                last_sl_idx = k
                break
                
        if last_sl_idx == -1: return None
        
        # Dernier Swing High avant ce Low
        last_sh_price = -1.0
        last_sh_idx = -1
        
        for k in range(last_sl_idx - 1, last_sl_idx - 100, -1):
            if rates[k]['is_swing_high']:
                last_sh_price = rates[k]['high']
                last_sh_idx = k
                break
                
        if last_sh_idx != -1:
            # Calcul Fib
            range_size = last_sh_price - last_sl_price
            fib_entry = last_sl_price + (range_size * FIB_LEVEL)
            sl_price = last_sh_price
            
            # FILTRE EMA : On ne vend pas si l'entrée est AU-DESSUS de l'EMA 200
            if fib_entry > ema_value:
                return None
            
            if current_close < fib_entry:
                tp_price = fib_entry - (abs(sl_price - fib_entry) * float(RR_TARGET))
                return {
                    "side": "SHORT",
                    "entry": fib_entry,
                    "sl": sl_price,
                    "tp": tp_price
                }

    return None

# -----------------------------------------------------------
# SIMULATION STRICTE (LIMIT ORDER)
# -----------------------------------------------------------

def run_simulation_strict(ltf_data, start_idx, setup, expiration_ts):
    is_filled = False
    entry = float(setup['entry'])
    sl = float(setup['sl'])
    tp = float(setup['tp'])
    side = setup['side']
    
    limit = min(start_idx + 10000, len(ltf_data))
    
    for i in range(start_idx, limit):
        row = ltf_data[i]
        ts, h, l = row['ts'], float(row['high']), float(row['low'])
        
        if ts > expiration_ts: return "EXPIRED", ts
        
        if not is_filled:
            # CHECK FILL
            if side == "LONG":
                if l <= entry: 
                    is_filled = True
                    if l <= sl: return "LOSS", ts
            elif side == "SHORT":
                if h >= entry:
                    is_filled = True
                    if h >= sl: return "LOSS", ts
            
        else: # TRADE OPEN
            if side == "LONG":
                if l <= sl: return "LOSS", ts
                if h >= tp: return "WIN", ts
            elif side == "SHORT":
                if h >= sl: return "LOSS", ts
                if l <= tp: return "WIN", ts

    return "EXPIRED", ltf_data[limit-1]['ts'] if ltf_data else expiration_ts

def execute_backtest(engine, pair, start_ms):
    table = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(SCAN_TF)}"
    try: df = pd.read_sql(f"SELECT ts as time, open, high, low, close FROM {table} ORDER BY ts ASC", engine)
    except: return []
    if df.empty: return []
    
    df = add_indicators(df)
    htf_rates = df.to_dict('records')
    
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_1m"
    try: df_ltf = pd.read_sql(f"SELECT ts, high, low FROM {table_ltf} ORDER BY ts ASC", engine)
    except: return []
    if df_ltf.empty: return []
    ltf_data = df_ltf.to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data]

    scan_ms = 15 * 60 * 1000
    trades = []
    skip_until = 0
    
    start_idx = 250
    if start_ms:
        for idx, r in enumerate(htf_rates):
            if r['time'] >= start_ms: start_idx = max(idx, 250); break

    for i in range(start_idx, len(htf_rates)):
        row = htf_rates[i]
        if row['time'] < skip_until: continue
        
        setup = detect_ema_fib_setup(htf_rates, i)
        
        if setup:
            sim_start = row['time'] + scan_ms
            l_idx = bisect.bisect_left(ltf_ts, sim_start)
            if l_idx >= len(ltf_data): continue
            
            expire = sim_start + (MAX_WAIT_CANDLES * scan_ms)
            
            res, exit_ts = run_simulation_strict(ltf_data, l_idx, setup, expire)
            
            if res in ["WIN", "LOSS"]:
                pnl_r = (RR_TARGET if res == "WIN" else Decimal("-1.0")) - FEES_PCT
                trades.append({"pair": pair, "result": res, "pnl_r": pnl_r, "exit_time": exit_ts})
                skip_until = exit_ts 

    return trades

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--start-date", type=str)
    args = ap.parse_args()
    
    start_ms = parse_date_to_ms(args.start_date) if args.start_date else None
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    
    print(f"--- EMA 200 + FIB 61.8 (GOLDEN TREND) ---")
    print(f"{'PAIRE':<10} {'TRADES':<8} {'WR':<8} {'PF':<8} {'PNL(R)':<10}")
    print("-" * 50)
    
    for p in pairs:
        t = execute_backtest(engine, p, start_ms)
        if not t: continue
        
        wins = sum(1 for x in t if x['result'] == "WIN")
        tot = len(t)
        wr = (wins/tot*100) if tot else 0
        prof = sum(x['pnl_r'] for x in t if x['pnl_r']>0)
        loss = sum(abs(x['pnl_r']) for x in t if x['pnl_r']<0)
        pf = prof/loss if loss > 0 else 0
        net = prof - loss
        
        print(f"{p:<10} {tot:<8} {wr:.1f}%   {pf:.2f}     {net:.2f}R")

if __name__ == "__main__":
    main()