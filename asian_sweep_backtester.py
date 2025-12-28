#!/usr/bin/env python3
# postgres_asian_range_sweep.py

import os
import re
import csv
import sys
import argparse
import statistics
import math
import bisect 
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIGURATION ASIAN SWEEP ----------
# Heures UTC
ASIAN_START_HOUR = 0
ASIAN_END_HOUR = 8      # Le range est défini de 00:00 à 08:00 exclu
TRADING_START_HOUR = 8  # On commence à chercher les setups à 08:00
TRADING_END_HOUR = 18   # On arrête de prendre des trades après 18:00

# Paramètres de Trade
SCAN_TF = "15m"
MAX_WAIT_CANDLES = 16   # 4 heures max pour que le trade se réalise
DEFAULT_RISK = Decimal("0.01") # 1% par trade
FEES_PCT = Decimal("0.10")     # Impact Spread/Comm sur R

# Filtres
MIN_RANGE_PIPS = 0      # Pas de filtre de taille pour l'instant
MAX_TRADES_PER_DAY = 1  # UN SEUL trade par jour (le premier bon signal) pour éviter l'overtrading

# -----------------------------------------------------------
# UTILS & DATABASE
# -----------------------------------------------------------

def price_scale(base: str, quote: str) -> int:
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime('%m-%d %H:%M')

def parse_date_to_ms(date_str: str, is_end_date: bool = False) -> int:
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT).replace(tzinfo=UTC)
        if is_end_date: dt += timedelta(days=1) - timedelta(milliseconds=1)
        return int(dt.timestamp() * 1000)
    except ValueError: raise ValueError(f"Format date invalide: {DATE_FORMAT}")

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    return create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")

def parse_pairs(path: str):
    out = []
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p: out.append(p.strip())
    return out

# -----------------------------------------------------------
# CALCUL DU RANGE ASIATIQUE
# -----------------------------------------------------------

def calculate_asian_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifie le High/Low entre 00:00 et 08:00 pour chaque jour.
    """
    df_calc = df.copy()
    df_calc['datetime'] = pd.to_datetime(df_calc['time'], unit='ms', utc=True)
    df_calc['date_key'] = df_calc['datetime'].dt.date
    df_calc['hour'] = df_calc['datetime'].dt.hour
    
    # Filtrer uniquement la session asiatique
    asian_session = df_calc[(df_calc['hour'] >= ASIAN_START_HOUR) & (df_calc['hour'] < ASIAN_END_HOUR)]
    
    # Grouper par jour pour trouver H/L
    daily_stats = asian_session.groupby('date_key').agg(
        asian_high=('high', 'max'),
        asian_low=('low', 'min')
    ).reset_index()
    
    # Fusionner avec le DF principal
    # Chaque bougie de la journée connaîtra le range asiatique de ce jour-là
    df_merged = pd.merge(df_calc, daily_stats, on='date_key', how='left')
    
    return df_merged

# -----------------------------------------------------------
# DETECTION DU SWEEP (FAKEOUT)
# -----------------------------------------------------------

def detect_asian_sweep(row, scale):
    """
    Vérifie si la bougie actuelle fait un Fakeout du range Asiatique.
    """
    ah = row['asian_high']
    al = row['asian_low']
    
    # Si pas de range calculé (ex: données manquantes ou dimanche soir), on passe
    if pd.isna(ah) or pd.isna(al): return None
    
    open_p, high, low, close = row['open'], row['high'], row['low'], row['close']
    
    setup = None
    
    # --- SCENARIO SHORT (SWEEP HIGH) ---
    # Le prix est allé au-dessus du Asian High, mais a clôturé EN DESSOUS.
    # C'est un rejet.
    if high > ah and close < ah:
        side = "SHORT"
        entry = Decimal(str(close))
        sl = Decimal(str(high)) # SL sur la mèche du fakeout
        tp = Decimal(str(al))   # Target le bas du range asiatique
        
        setup = {"side": side, "entry": qround(entry, scale), "sl": qround(sl, scale), "tp": qround(tp, scale)}

    # --- SCENARIO LONG (SWEEP LOW) ---
    # Le prix est allé en-dessous du Asian Low, mais a clôturé AU-DESSUS.
    elif low < al and close > al:
        side = "LONG"
        entry = Decimal(str(close))
        sl = Decimal(str(low))  # SL sur la mèche du fakeout
        tp = Decimal(str(ah))   # Target le haut du range asiatique
        
        setup = {"side": side, "entry": qround(entry, scale), "sl": qround(sl, scale), "tp": qround(tp, scale)}
        
    return setup

# -----------------------------------------------------------
# MOTEUR DE SIMULATION
# -----------------------------------------------------------

def run_simulation(ltf_data, start_idx, setup, expiration_ts):
    is_open = False
    entry, sl, tp, side = float(setup['entry']), float(setup['sl']), float(setup['tp']), setup['side']
    
    limit = min(start_idx + 5000, len(ltf_data))
    
    for i in range(start_idx, limit):
        row = ltf_data[i]
        ts, h, l = row['ts'], float(row['high']), float(row['low'])
        
        if not is_open:
            if ts > expiration_ts: return "EXPIRED", ts
            
            # Entrée Market à la clôture de la bougie précédente (donc Open de celle-ci)
            # On considère qu'on est fill immédiatement
            is_open = True
            
            # Check immédiat SL sur la même bougie (si volatilité extrême)
            if side == "LONG" and l <= sl: return "LOSS", ts
            if side == "SHORT" and h >= sl: return "LOSS", ts
        
        if is_open:
            if side == "LONG":
                if l <= sl: return "LOSS", ts
                if h >= tp: return "WIN", ts
            elif side == "SHORT":
                if h >= sl: return "LOSS", ts
                if l <= tp: return "WIN", ts
                
    return "EXPIRED", ltf_data[limit-1]['ts'] if ltf_data else expiration_ts

def execute_backtest(engine, pair, start_ms, end_ms):
    # 1. FETCH HTF (15m)
    table = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(SCAN_TF)}"
    query = f"SELECT ts as time, open, high, low, close FROM {table} ORDER BY ts ASC"
    try: df = pd.read_sql(query, engine)
    except: return []
    if df.empty: return []
    
    # 2. CALCUL RANGES
    df = calculate_asian_ranges(df)
    htf_rates = df.to_dict('records')

    # 3. FETCH LTF (1m)
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_1m"
    try: df_ltf = pd.read_sql(f"SELECT ts, high, low FROM {table_ltf} ORDER BY ts ASC", engine)
    except: return []
    if df_ltf.empty: return []
    ltf_data = df_ltf.to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data]

    scale = price_scale(pair[:3], pair[3:])
    scan_ms = 15 * 60 * 1000 # 15m
    
    trades = []
    daily_trade_count = {} # Pour limiter à 1 trade par jour
    
    start_idx = 0
    if start_ms:
        for idx, r in enumerate(htf_rates):
            if r['time'] >= start_ms: start_idx = idx; break

    skip_until = 0

    # 4. BOUCLE
    for i in range(start_idx, len(htf_rates)):
        row = htf_rates[i]
        if end_ms and row['time'] > end_ms: break
        if row['time'] < skip_until: continue
        
        # Vérification Heure de Trading
        hour = row['hour']
        if not (TRADING_START_HOUR <= hour < TRADING_END_HOUR):
            continue
            
        # Vérification limite journalière
        date_key = row['date_key']
        if daily_trade_count.get(date_key, 0) >= MAX_TRADES_PER_DAY:
            continue
            
        setup = detect_asian_sweep(row, scale)
        
        if setup:
            # Calcul RR prévisionnel
            risk = abs(setup['entry'] - setup['sl'])
            reward = abs(setup['tp'] - setup['entry'])
            if risk == 0: continue
            rr = reward / risk
            
            # Filtre RR Minime (évite les trades inutiles)
            if rr < 0.8: continue 
            
            sim_start = row['time'] + scan_ms
            l_idx = bisect.bisect_left(ltf_ts, sim_start)
            if l_idx >= len(ltf_data): continue
            
            expire = sim_start + (MAX_WAIT_CANDLES * scan_ms)
            
            res, exit_ts = run_simulation(ltf_data, l_idx, setup, expire)
            
            if res in ["WIN", "LOSS"]:
                pnl_r = (rr if res == "WIN" else Decimal("-1.0")) - FEES_PCT
                
                trades.append({
                    "pair": pair, "side": setup['side'], "result": res, 
                    "pnl_r": pnl_r, "rr": rr,
                    "entry_time": row['time'], "exit_time": exit_ts
                })
                
                daily_trade_count[date_key] = daily_trade_count.get(date_key, 0) + 1
                skip_until = exit_ts # On attend la fin du trade avant de scanner

    return trades

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt") # Utilise ton fichier complet
    ap.add_argument("--start-date", type=str)
    args = ap.parse_args()
    
    start_ms = parse_date_to_ms(args.start_date) if args.start_date else None
    
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    print(f"--- ASIAN RANGE SWEEP (London/NY Open) ---")
    print(f"Asian Session: {ASIAN_START_HOUR}:00 - {ASIAN_END_HOUR}:00 UTC")
    print(f"Trading Window: {TRADING_START_HOUR}:00 - {TRADING_END_HOUR}:00 UTC")
    
    all_trades = []
    
    print(f"{'PAIRE':<10} {'TRADES':<8} {'WR':<8} {'PF':<8} {'PNL(R)':<10}")
    print("-" * 50)
    
    global_results = []
    
    for p in pairs:
        t = execute_backtest(engine, p, start_ms, None)
        if not t: continue
        
        all_trades.extend(t)
        
        wins = sum(1 for x in t if x['result'] == "WIN")
        tot = len(t)
        wr = (wins/tot*100) if tot else 0
        prof = sum(x['pnl_r'] for x in t if x['pnl_r']>0)
        loss = sum(abs(x['pnl_r']) for x in t if x['pnl_r']<0)
        pf = prof/loss if loss > 0 else 0
        net = prof - loss
        
        global_results.append({'pair': p, 'pf': pf, 'net': net, 'wr': wr, 'trades': tot})
        print(f"{p:<10} {tot:<8} {wr:.1f}%   {pf:.2f}     {net:.2f}R")

    # GLOBAL STATS
    if all_trades:
        all_trades.sort(key=lambda x: x['exit_time'])
        balance = Decimal("50000.00")
        initial = balance
        
        for t in all_trades:
            risk_amt = balance * DEFAULT_RISK
            pnl_usd = risk_amt * t['pnl_r']
            balance += pnl_usd
            
        print("\n" + "="*40)
        print(f"FINAL BALANCE: {balance:,.2f} USD")
        print(f"ROI: {((balance-initial)/initial)*100:.2f}%")
        print("="*40)
        
        # TOP PERFORMERS
        global_results.sort(key=lambda x: x['pf'], reverse=True)
        print("\nTOP 5 PAIRES (PROFIT FACTOR):")
        for r in global_results[:5]:
            print(f"{r['pair']}: PF {r['pf']:.2f} | Net {r['net']:.2f}R")

if __name__ == "__main__":
    main()