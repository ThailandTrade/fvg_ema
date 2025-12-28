#!/usr/bin/env python3
# postgres_smc_structure_15m.py

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

# ---------- CONFIGURATION SMC "ULTIMATE" ----------
SCAN_TF = "15m"                    # Timeframe de Structure
EXECUTION_TF_SUFFIX = "1m"         # Pour la précision du touch

# Paramètres de Structure
SWING_LOOKBACK = 40                # Combien de bougies on regarde pour définir le High/Low majeur
MIN_RETRACTION = 0.50              # 0.50 = Equilibrium (On ne trade que si retracement > 50%)

# Gestion du Risque
RISK_PER_TRADE = Decimal("0.01")   # 1% par trade
FEES_PCT = Decimal("0.05")         # Frais réduits (compte Raw)
INITIAL_BALANCE = Decimal("50000.00")

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
# CŒUR DU REACTEUR : LOGIQUE SMC
# -----------------------------------------------------------

def find_best_fvg_in_zone(rates, start_idx, end_idx, direction, zone_limit):
    """
    Scanne l'historique (l'impulsion) pour trouver le meilleur FVG situé dans la bonne zone.
    direction "UP" (Bullish Impulse) -> Cherche Bullish FVG < zone_limit (Discount)
    direction "DOWN" (Bearish Impulse) -> Cherche Bearish FVG > zone_limit (Premium)
    """
    best_fvg_price = None
    max_gap_size = -1.0
    
    # On parcourt l'impulsion pour trouver les traces laissées
    for k in range(start_idx + 2, end_idx):
        row = rates[k]
        prev2 = rates[k-2]
        
        # BULLISH FVG (L'impulsion était haussière)
        if direction == "UP":
            # Condition FVG : Low[k] > High[k-2]
            if row['low'] > prev2['high']:
                gap_size = row['low'] - prev2['high']
                fvg_entry = prev2['high'] # On entre au début du FVG (le haut de la bougie 1)
                
                # FILTRE CRUCIAL : Est-ce que ce FVG est en DISCOUNT ?
                if fvg_entry < zone_limit:
                    # On garde le plus gros FVG trouvé dans la zone
                    if gap_size > max_gap_size:
                        max_gap_size = gap_size
                        best_fvg_price = fvg_entry # Limit Order ici

        # BEARISH FVG (L'impulsion était baissière)
        elif direction == "DOWN":
            # Condition FVG : High[k] < Low[k-2]
            if row['high'] < prev2['low']:
                gap_size = prev2['low'] - row['high']
                fvg_entry = prev2['low'] # On entre au début du FVG (le bas de la bougie 1)
                
                # FILTRE CRUCIAL : Est-ce que ce FVG est en PREMIUM ?
                if fvg_entry > zone_limit:
                    if gap_size > max_gap_size:
                        max_gap_size = gap_size
                        best_fvg_price = fvg_entry

    return best_fvg_price

def detect_smc_setup(rates, i, scale):
    """
    Identifie le Dealing Range, l'Equilibrium, et cherche un FVG historique à targeter.
    """
    if i < SWING_LOOKBACK: return None
    
    # 1. Identifier le Dealing Range (Swing High / Swing Low)
    window = rates[i-SWING_LOOKBACK : i] # On ne regarde pas la bougie actuelle pour définir le range (lag naturel)
    
    # On cherche les indices relatifs dans la fenêtre
    highs = [r['high'] for r in window]
    lows = [r['low'] for r in window]
    
    max_h = max(highs)
    min_l = min(lows)
    
    # Indices absolus dans 'rates'
    idx_max_h = (i - SWING_LOOKBACK) + highs.index(max_h)
    idx_min_l = (i - SWING_LOOKBACK) + lows.index(min_l)
    
    side = ""
    entry = Decimal(0); sl = Decimal(0); tp = Decimal(0)
    
    # 2. Analyse de la Structure
    
    # --- STRUCTURE HAUSSIÈRE (Bullish Dealing Range) ---
    # Le Low est ancien, le High est récent -> On a monté.
    if idx_min_l < idx_max_h:
        # On est en tendance haussière sur ce timeframe
        dealing_range_low = min_l
        dealing_range_high = max_h
        equilibrium = (dealing_range_low + dealing_range_high) / 2
        
        # CONDITION 1 : Le prix actuel doit être revenu en DISCOUNT (< Eq)
        current_low = rates[i]['low']
        if current_low < equilibrium:
            
            # CONDITION 2 : La structure doit tenir (ne pas casser le Low Majeur)
            if current_low > dealing_range_low:
                
                # CONDITION 3 : Trouver un FVG dans la jambe de hausse (Entre Low et High)
                # qui se trouve en zone Discount.
                fvg_entry_level = find_best_fvg_in_zone(rates, idx_min_l, idx_max_h, "UP", equilibrium)
                
                if fvg_entry_level:
                    # TRIGGER : Le prix actuel vient-il de toucher ce niveau ?
                    if current_low <= fvg_entry_level:
                        side = "LONG"
                        entry = Decimal(str(fvg_entry_level)) # Limit order simulé
                        sl = Decimal(str(dealing_range_low))  # SL sous le Swing Low (Invalidation)
                        tp = Decimal(str(dealing_range_high)) # TP sur le Swing High (Liquidité)

    # --- STRUCTURE BAISSIÈRE (Bearish Dealing Range) ---
    # Le High est ancien, le Low est récent -> On a baissé.
    elif idx_max_h < idx_min_l:
        dealing_range_high = max_h
        dealing_range_low = min_l
        equilibrium = (dealing_range_high + dealing_range_low) / 2
        
        # CONDITION 1 : Le prix actuel doit être remonté en PREMIUM (> Eq)
        current_high = rates[i]['high']
        if current_high > equilibrium:
            
            # CONDITION 2 : La structure doit tenir (ne pas casser le High Majeur)
            if current_high < dealing_range_high:
                
                # CONDITION 3 : Trouver un FVG dans la jambe de baisse qui est en Premium
                fvg_entry_level = find_best_fvg_in_zone(rates, idx_max_h, idx_min_l, "DOWN", equilibrium)
                
                if fvg_entry_level:
                    # TRIGGER : Le prix actuel touche le niveau ?
                    if current_high >= fvg_entry_level:
                        side = "SHORT"
                        entry = Decimal(str(fvg_entry_level))
                        sl = Decimal(str(dealing_range_high)) # SL au-dessus du Swing High
                        tp = Decimal(str(dealing_range_low))  # TP sur le Swing Low

    if not side: return None
    
    # Calcul du RR pour info
    dist_sl = abs(entry - sl)
    dist_tp = abs(tp - entry)
    if dist_sl == 0: return None
    rr = dist_tp / dist_sl
    
    # Filtre RR minimum (on ne prend pas si le RR est < 1.5 par exemple, SMC vise souvent 3+)
    if rr < 1.5: return None 

    return {
        "side": side,
        "entry": qround(entry, scale),
        "sl": qround(sl, scale),
        "tp": qround(tp, scale),
        "rr": rr
    }

# -----------------------------------------------------------
# MOTEUR D'EXECUTION
# -----------------------------------------------------------

def run_simulation(ltf_data, start_idx, setup, expiration_ts):
    is_open = False
    entry, sl, tp, side = float(setup['entry']), float(setup['sl']), float(setup['tp']), setup['side']
    
    # On cherche l'exécution précise
    limit = min(start_idx + 5000, len(ltf_data))
    
    for i in range(start_idx, limit):
        row = ltf_data[i]
        ts, h, l = row['ts'], float(row['high']), float(row['low'])
        
        if not is_open:
            if ts > expiration_ts: return "EXPIRED", ts
            
            # ENTRY LOGIC (LIMIT ORDER)
            # On assume qu'on est rempli si le prix traverse notre niveau
            if (side == "LONG" and l <= entry):
                is_open = True
                # Check immédiat SL dans la même bougie (mauvais signe)
                if l <= sl: return "LOSS", ts
                
            elif (side == "SHORT" and h >= entry):
                is_open = True
                if h >= sl: return "LOSS", ts
        
        if is_open:
            if side == "LONG":
                if l <= sl: return "LOSS", ts
                if h >= tp: return "WIN", ts
            elif side == "SHORT":
                if h >= sl: return "LOSS", ts
                if l <= tp: return "WIN", ts
                
    return "EXPIRED", ltf_data[limit-1]['ts'] if ltf_data else expiration_ts

def execute_backtest(engine, pair, start_ms, end_ms):
    # Fetch 15m Data
    table = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(SCAN_TF)}"
    query = f"SELECT ts as time, open, high, low, close FROM {table} ORDER BY ts ASC"
    try: df = pd.read_sql(query, engine)
    except: return []
    if df.empty: return []
    
    htf_rates = df.to_dict('records')

    # Fetch 1m Data
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_1m"
    try: df_ltf = pd.read_sql(f"SELECT ts, high, low FROM {table_ltf} ORDER BY ts ASC", engine)
    except: return []
    if df_ltf.empty: return []
    ltf_data = df_ltf.to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data]

    scale = price_scale(pair[:3], pair[3:])
    scan_ms = htf_rates[1]['time'] - htf_rates[0]['time']
    skip_until = 0
    trades = []
    
    start_idx = SWING_LOOKBACK + 5
    if start_ms:
        for idx, r in enumerate(htf_rates):
            if r['time'] >= start_ms: 
                start_idx = max(idx, start_idx); break

    # MAIN LOOP
    for i in range(start_idx, len(htf_rates)):
        row = htf_rates[i]
        if end_ms and row['time'] > end_ms: break
        if row['time'] < skip_until: continue
        
        setup = detect_smc_setup(htf_rates, i, scale)
        
        if setup:
            # On simule à partir de la bougie suivante
            sim_start = row['time'] + scan_ms
            l_idx = bisect.bisect_left(ltf_ts, sim_start)
            
            if l_idx >= len(ltf_data): continue
            
            # On laisse le trade vivre assez longtemps (Swing trade intraday)
            # 48 * 15m = 12 heures
            expire = sim_start + (48 * scan_ms)
            
            res, exit_ts = run_simulation(ltf_data, l_idx, setup, expire)
            
            if res in ["WIN", "LOSS"]:
                # PnL Calculation
                # Win = RR * Risk (ex: 3R * 1%)
                # Loss = -1 * Risk
                pnl_r = (setup['rr'] if res == "WIN" else Decimal("-1.0")) - FEES_PCT
                
                trades.append({
                    "pair": pair, "side": setup['side'], "result": res, 
                    "pnl_r": pnl_r, "rr": setup['rr'],
                    "entry_time": row['time'], "exit_time": exit_ts
                })
                skip_until = exit_ts

    return trades

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs_portfolio.txt")
    ap.add_argument("--start-date", type=str)
    args = ap.parse_args()
    
    start_ms = parse_date_to_ms(args.start_date) if args.start_date else None
    
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    print(f"--- SMC STRUCTURE TRADER (TF: {SCAN_TF}) ---")
    print(f"Scanning {len(pairs)} pairs for Dealing Ranges & FVGs...")
    
    all_trades = []
    
    print(f"{'PAIRE':<10} {'TRADES':<8} {'WR':<8} {'PF':<8} {'PNL(R)':<10}")
    print("-" * 50)
    
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
        
        print(f"{p:<10} {tot:<8} {wr:.1f}%   {pf:.2f}     {net:.2f}R")

    # GLOBAL STATS
    if all_trades:
        all_trades.sort(key=lambda x: x['exit_time'])
        balance = INITIAL_BALANCE
        
        for t in all_trades:
            # Risk Management dynamique : % de la balance courante
            risk_amt = balance * RISK_PER_TRADE
            pnl_usd = risk_amt * t['pnl_r']
            balance += pnl_usd
            
        print("\n" + "="*40)
        print(f"FINAL BALANCE: {balance:,.2f} USD")
        print(f"ROI: {((balance-INITIAL_BALANCE)/INITIAL_BALANCE)*100:.2f}%")
        print("="*40)

if __name__ == "__main__":
    main()