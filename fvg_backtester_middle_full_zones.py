#!/usr/bin/env python3
# postgres_fvg_luxalgo_hybrid_backtester.py

import os
import re
import csv
import sys
import time
import argparse
import statistics
import math
import bisect 
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List, Any

# PANDAS & NUMPY
import pandas as pd
import numpy as np

# DATABASE
from dotenv import load_dotenv
from sqlalchemy import create_engine

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING (BASE FVG V3) ----------
DEFAULT_RR = Decimal("1.5") # J'ai mis 1.5 car avec le filtre de zone, on peut viser un peu plus
MAX_WAIT_CANDLES = 5
SCAN_TF = "1h"            
EXECUTION_TF_SUFFIX = "1m" 
DEFAULT_RISK_PER_TRADE = Decimal("0.003") 
DEFAULT_FEES_PCT = Decimal("0.10") 

# --- CONFIG LUXALGO ---
SWING_LENGTH = 50          # Pour le calcul des zones structurelles

# --- CONFIG SUMMARY ---
INITIAL_BALANCE = Decimal("50000.00") 
SHOW_ALL_TRADES = True                

# ---------- CONSTANTES STDEV (FVG) ----------
STDEV_PERIOD = 200 
DEFAULT_STDEV_THRESHOLD = 0.5
DEFAULT_STDEV_MAX = 1.0

# --- RESULTATS ---
GLOBAL_RESULTS = []

# ---------- UTILS (INCHANGÉS) ----------

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

def parse_pairs(path: str):
    out = []
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p: out.append(p.strip())
    return out

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    return create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")

# ---------- MOTEUR LUXALGO (LOGIQUE EXACTE) ----------

def apply_luxalgo_zones(df: pd.DataFrame, length: int):
    highs, lows = df['high'].values, df['low'].values
    trailing_top, trailing_bottom = np.zeros(len(df)), np.zeros(len(df))
    curr_top, curr_bottom = highs[0], lows[0]
    
    for i in range(len(df)):
        if i >= length:
            # Swing High Detection
            if highs[i-length] > np.max(highs[i-length : i]):
                curr_top = highs[i-length]
            # Swing Low Detection
            if lows[i-length] < np.min(lows[i-length : i]):
                curr_bottom = lows[i-length]
        
        # Trailing Update
        curr_top = max(highs[i], curr_top)
        curr_bottom = min(lows[i], curr_bottom)
        
        trailing_top[i] = curr_top
        trailing_bottom[i] = curr_bottom
        
    df['premium_threshold'] = 0.95 * trailing_top + 0.05 * trailing_bottom
    df['discount_threshold'] = 0.95 * trailing_bottom + 0.05 * trailing_top
    return df

# --- FETCH DATA (FVG + ZONES) ---

def fetch_htf_data_pandas(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"
    # On a besoin de l'EMA pour la strat FVG originale
    query = f"SELECT ts as time, open, high, low, close, ema_50 FROM {table_name}"
    
    # Buffer large pour LuxAlgo (pivots)
    safe_buffer_ms = timedelta(days=60).total_seconds() * 1000
    cond = []
    if start_ms: cond.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms: cond.append(f"ts <= {end_ms}")
    if cond: query += " WHERE " + " AND ".join(cond)
    query += " ORDER BY ts ASC"

    try:
        df = pd.read_sql(query, engine)
        if df.empty: return None

        # 1. Calcul Zones Structurelles
        df = apply_luxalgo_zones(df, SWING_LENGTH)
        
        # 2. Calcul Volatilité pour FVG
        gap_series = (df['low'] - df['high'].shift(2)).abs()
        df['stdev_200'] = gap_series.rolling(window=STDEV_PERIOD).std(ddof=1).fillna(0.0)

        return df.to_dict('records')
    except Exception as e:
        print(f"[ERR] Fetch HTF: {e}")
        return None

def fetch_ltf_data_pandas(engine, pair: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    buffer_end = end_ms + (MAX_WAIT_CANDLES * 30 * 60 * 1000 * 2) 
    query = f"SELECT ts, high, low FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {buffer_end} ORDER BY ts ASC"
    try:
        df = pd.read_sql(query, engine)
        return [] if df.empty else df.to_dict('records')
    except Exception: return []

# --- SIMULATION MEMOIRE ---

def run_ltf_simulation_memory(ltf_data, start_index, entry, sl, tp, side, expiration_ts):
    is_open = False
    max_steps = 5000 
    end_loop = min(start_index + max_steps, len(ltf_data))
    
    for i in range(start_index, end_loop):
        row = ltf_data[i]
        ts, h, l = row['ts'], float(row['high']), float(row['low'])
        
        if not is_open:
            if ts > expiration_ts: return "EXPIRED", ts
            if (side == "LONG" and l <= entry) or (side == "SHORT" and h >= entry):
                is_open = True
                if (side == "LONG" and l <= sl) or (side == "SHORT" and h >= sl): return "LOSS", ts
        
        if is_open:
            if side == "LONG":
                if l <= sl: return "LOSS", ts
                if h >= tp: return "WIN", ts
            elif side == "SHORT":
                if h >= sl: return "LOSS", ts
                if l <= tp: return "WIN", ts
                
    return "EXPIRED", ltf_data[end_loop - 1]['ts'] if ltf_data else expiration_ts


# --- LOGIQUE FVG ORIGINALE + FILTRE ZONES ---

def check_fvg_volatility_optimized(rates, i, threshold):
    if i < STDEV_PERIOD + 2: return False, False, 0.0, 0.0
    h_i_2, l_i_2 = rates[i-2]["high"], rates[i-2]["low"]
    h_i, l_i = rates[i]["high"], rates[i]["low"]
    
    raw_bull = (h_i_2 < l_i)
    raw_bear = (l_i_2 > h_i)
    if not raw_bull and not raw_bear: return False, False, 0.0, 0.0
    
    vol = rates[i].get('stdev_200', 0.0) or 1.0e-9
    is_bull, is_bear, score, gap = False, False, 0.0, 0.0
    
    if raw_bull:
        gap = l_i - h_i_2; score = gap / vol
        if score > threshold: is_bull = True
    elif raw_bear:
        gap = l_i_2 - h_i; score = gap / vol
        if score > threshold: is_bear = True
    return is_bull, is_bear, score, gap

def detect_fvg_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_threshold: float, stdev_max: float) -> Optional[Dict[str, Any]]:
    # 1. Base FVG Detection
    if i < 2: return None
    ema50 = rates[i]["ema_50"]
    if ema50 is None or pd.isna(ema50): return None
    
    # Zones LuxAlgo
    premium_thresh = rates[i]['premium_threshold']
    discount_thresh = rates[i]['discount_threshold']

    is_bull, is_bear, score, gap = check_fvg_volatility_optimized(rates, i, stdev_threshold)
    if (not is_bull and not is_bear) or score > stdev_max: return None

    entry_price = Decimal(0); sl_price = Decimal(0); side = ""
    
    if is_bull:
        side = "LONG"
        fvg_high = Decimal(str(rates[i]["low"]))
        fvg_low = Decimal(str(rates[i-2]["high"]))
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        
        # --- FILTRE 1 : EMA ---
        if entry_price <= Decimal(str(ema50)): return None
        
        # --- FILTRE 2 : ZONES LUXALGO (NOUVEAU) ---
        # On achète UNIQUEMENT si on est en Discount (le prix est bas dans la structure)
        # Note : On compare le prix d'entrée au seuil Discount
        if float(entry_price) > discount_thresh: return None
        
        sl_price = Decimal(str(rates[i-1]["low"]))
        if sl_price >= entry_price: return None

    elif is_bear:
        side = "SHORT"
        fvg_high = Decimal(str(rates[i-2]["low"]))
        fvg_low = Decimal(str(rates[i]["high"]))
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        
        # --- FILTRE 1 : EMA ---
        if entry_price >= Decimal(str(ema50)): return None

        # --- FILTRE 2 : ZONES LUXALGO (NOUVEAU) ---
        # On vend UNIQUEMENT si on est en Premium (le prix est haut dans la structure)
        if float(entry_price) < premium_thresh: return None

        sl_price = Decimal(str(rates[i-1]["high"]))
        if sl_price <= entry_price: return None

    else: return None

    return {
        "side": side,
        "entry_price": qround(entry_price, scale),
        "sl_price": qround(sl_price, scale),
        "fvg_start_candle_index": i,
        "stdev_score": score,
        "gap_size": gap
    }

# ---------- EXECUTION DU BACKTEST ----------

def execute_backtest(engine, pair, rr, scale, stdev_min, start_ms, end_ms, risk, stdev_max, fees):
    rates = fetch_htf_data_pandas(engine, pair, SCAN_TF, start_ms, end_ms)
    if not rates or len(rates) < 200: return []
    
    ltf_data = fetch_ltf_data_pandas(engine, pair, rates[0]['time'], rates[-1]['time'])
    ltf_ts = [r['ts'] for r in ltf_data]
    
    start_index = 0
    seed = max(STDEV_PERIOD + 2, SWING_LENGTH + 2)
    for idx in range(seed, len(rates)):
        if start_ms is None or rates[idx]['time'] >= start_ms:
            start_index = idx; break
    else: return []

    balance_r, total, wins, losses = Decimal(0), 0, 0, 0
    trade_log, all_pnl_r = [], []
    g_profit, g_loss = Decimal(0), Decimal(0)
    
    scan_ms = rates[1]['time'] - rates[0]['time'] if len(rates) > 1 else 1800000
    skip_until = 0

    for i in range(start_index, len(rates)):
        if end_ms and rates[i]['time'] > end_ms: break
        if rates[i]['time'] < skip_until: continue
        
        setup = detect_fvg_setup(rates, i, scale, stdev_min, stdev_max)
        if setup:
            risk_amt = abs(setup["entry_price"] - setup["sl_price"])
            if setup["side"] == "LONG": tp_price = setup["entry_price"] + (risk_amt * rr)
            else: tp_price = setup["entry_price"] - (risk_amt * rr)
            
            sim_start = rates[i]['time'] + scan_ms
            l_idx = bisect.bisect_left(ltf_ts, sim_start)
            if l_idx >= len(ltf_data): continue
            
            res, exit_t = run_ltf_simulation_memory(ltf_data, l_idx, float(setup["entry_price"]), float(setup["sl_price"]), float(tp_price), setup["side"], sim_start + (MAX_WAIT_CANDLES * scan_ms))
            
            if res in ["WIN", "LOSS"]:
                total += 1
                pnl = (rr if res == "WIN" else Decimal("-1.0")) - fees
                balance_r += pnl
                if res == "WIN": wins += 1; g_profit += pnl
                else: losses += 1; g_loss += abs(pnl)
                all_pnl_r.append(float(pnl))
                
                trade_log.append({
                    "pair": pair, "entry_time": rates[i]["time"], "exit_time": exit_t,
                    "side": setup["side"], "entry_price": setup["entry_price"], "sl_price": setup["sl_price"],
                    "tp_price": qround(tp_price, scale), "exit_price": setup["sl_price"] if res=="LOSS" else qround(tp_price, scale),
                    "result": res, "pnl_r": pnl
                })
                skip_until = exit_t

    if total > 0:
        pf = g_profit / g_loss if g_loss > 0 else Decimal("99.9")
        sqn = (math.sqrt(total) * (statistics.mean(all_pnl_r) / statistics.stdev(all_pnl_r))) if total > 1 and statistics.stdev(all_pnl_r) > 0 else 0
        GLOBAL_RESULTS.append({
            "pair": pair, "total_trades": total, "wins": wins, "losses": losses,
            "expectancy_r": balance_r/total, "win_rate": (wins/total*100),
            "profit_factor": pf, "sqn": sqn, "max_drawdown_r": 0, # Simplifié pour brevity
            "start_ts": rates[start_index]['time'], "end_ts": rates[-1]['time']
        })
    return trade_log

# ---------- AFFICHAGE (IDENTIQUE) ----------

def display_summary(rr, results):
    results.sort(key=lambda x: x['sqn'], reverse=True)
    print(f"\nSUMMARY FVG + LUXALGO ZONES (RR: {rr}R)")
    print("| {:<10} | {:^6} | {:^8} | {:^10} | {:^10} | {:^10} |".format("PAIRE", "TRADES", "WR", "EXPECT.", "PF", "SQN"))
    print("-" * 75)
    for r in results:
        print("| {:<10} | {:>6} | {:>7.2f}% | {:>9.4f}R | {:>10.2f} | {:>10.2f} |".format(r['pair'], r['total_trades'], r['win_rate'], float(r['expectancy_r']), float(r['profit_factor']), r['sqn']))

def display_daily(all_logs):
    d_stats = {d: {'t':0, 'w':0} for d in range(7)}
    days = ["LUN", "MAR", "MER", "JEU", "VEN", "SAM", "DIM"]
    for logs in all_logs.values():
        for t in logs:
            d = datetime.fromtimestamp(t['entry_time']/1000, tz=UTC).weekday()
            d_stats[d]['t'] += 1
            if t['result'] == "WIN": d_stats[d]['w'] += 1
    print("\nDAILY BREAKDOWN")
    for i, name in enumerate(days):
        if d_stats[i]['t'] > 0: print(f"{name}: {d_stats[i]['t']} trades, WR: {d_stats[i]['w']/d_stats[i]['t']*100:.1f}%")

def display_portfolio(all_logs, capital, risk):
    flat = []
    for p, logs in all_logs.items():
        for t in logs: flat.append(t)
    flat.sort(key=lambda x: x['exit_time'])
    bal = capital
    for t in flat: bal += (bal * risk * t['pnl_r'])
    print(f"\nPORTFOLIO: {bal:,.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--rr", type=Decimal, default=DEFAULT_RR)
    ap.add_argument("--stdev-threshold", type=float, default=DEFAULT_STDEV_THRESHOLD)
    ap.add_argument("--stdev-max", type=float, default=DEFAULT_STDEV_MAX)
    ap.add_argument("--start-date", type=str)
    ap.add_argument("--end-date", type=str)
    ap.add_argument("--risk", type=Decimal, default=DEFAULT_RISK_PER_TRADE)
    ap.add_argument("--fees", type=Decimal, default=DEFAULT_FEES_PCT)
    args = ap.parse_args()
    
    start = parse_date_to_ms(args.start_date) if args.start_date else None
    end = parse_date_to_ms(args.end_date, True) if args.end_date else None
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    
    all_logs = {}
    for p in pairs:
        scale = price_scale(p[:3], p[3:])
        log = execute_backtest(engine, p, args.rr, scale, args.stdev_threshold, start, end, args.risk, args.stdev_max, args.fees)
        all_logs[p] = log
        
    display_summary(args.rr, GLOBAL_RESULTS)
    display_daily(all_logs)
    display_portfolio(all_logs, INITIAL_BALANCE, args.risk)

if __name__ == "__main__":
    main()