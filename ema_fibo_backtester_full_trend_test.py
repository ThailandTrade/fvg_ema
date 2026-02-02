#!/usr/bin/env python3
# postgres_fvg_backtester_OPTIMIZED_RR.py

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

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING ----------
MAX_WAIT_CANDLES = 72
SCAN_TF = "5m"            
TREND_FILTER_TF = "1h"   
EXECUTION_TF_SUFFIX = "1m" 
DEFAULT_RISK_PER_TRADE = Decimal("0.001")
DEFAULT_FEES_PCT = Decimal("0.0")

# --- PARAMETRES STRATEGIE ---
EMA_TREND_PERIOD = 200
FIB_RETREACEMENT = 0.62
SWING_CONFIRMATION_LAG = 5 
INITIAL_BALANCE = Decimal("100000.00")

# --- OPTIMISATION ---
RR_STEPS = [Decimal(str(x)) for x in [1.5, 2.0, 2.5, 3.0]] # Valeurs de RR à tester

GLOBAL_RESULTS = []

# ---------- UTILS ----------

def price_scale(base: str, quote: str) -> int:
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime('%m-%d %H:%M')

def parse_date_to_ms(date_str: str, is_end_date: bool = False) -> int:
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT).replace(tzinfo=UTC)
        if is_end_date:
            dt += timedelta(days=1) - timedelta(milliseconds=1)
        return int(dt.timestamp() * 1000)
    except:
        raise ValueError(f"Format date: {DATE_FORMAT}")

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
    engine = create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")
    return engine

# ---------- DATA FETCHING (Inchangé pour la sécurité) ----------

def fetch_trend_structure_data(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> pd.DataFrame:
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"
    query = f"SELECT ts as time, high, low, close FROM {table_name}"
    safe_buffer_ms = timedelta(days=60).total_seconds() * 1000
    conds = []
    if start_ms: conds.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms: conds.append(f"ts <= {end_ms}")
    if conds: query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY ts ASC"
    try:
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()
        l = SWING_CONFIRMATION_LAG
        window = 2 * l + 1
        rolling_max, rolling_min = df['high'].rolling(window=window).max(), df['low'].rolling(window=window).min()
        df['is_swing_high'] = (df['high'] == rolling_max.shift(-l))
        df['is_swing_low'] = (df['low'] == rolling_min.shift(-l))
        last_h, last_l, prev_h, prev_l, current_trend = -1.0, -1.0, -1.0, -1.0, 0
        trends = []
        for idx, row in df.iterrows():
            if row['is_swing_high']: prev_h = last_h; last_h = row['high']
            if row['is_swing_low']: prev_l = last_l; last_l = row['low']
            if last_h > 0 and prev_h > 0 and last_l > 0 and prev_l > 0:
                if last_h > prev_h and last_l > prev_l: current_trend = 1
                elif last_l < prev_l and last_h < prev_h: current_trend = -1
            trends.append(current_trend)
        df['htf_trend'] = trends
        df['htf_trend'] = df['htf_trend'].shift(1).fillna(0)
        return df[['time', 'htf_trend']]
    except: return pd.DataFrame()

def fetch_htf_data_pandas_raw(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> pd.DataFrame:
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"
    query = f"SELECT ts as time, open, high, low, close FROM {table_name}"
    safe_buffer_ms = timedelta(days=40).total_seconds() * 1000
    conds = []
    if start_ms: conds.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms: conds.append(f"ts <= {end_ms}")
    if conds: query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY ts ASC"
    try:
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()
        df['ema_trend'] = df['close'].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
        l = SWING_CONFIRMATION_LAG
        window = 2 * l + 1
        df['is_swing_high'] = (df['high'] == df['high'].rolling(window=window, center=True).max())
        df['is_swing_low'] = (df['low'] == df['low'].rolling(window=window, center=True).min())
        return df
    except: return pd.DataFrame()

def fetch_ltf_data_pandas(engine, pair: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    query = f"SELECT ts, high, low FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {end_ms + 86400000} ORDER BY ts ASC"
    try:
        return pd.read_sql(query, engine).to_dict('records')
    except: return []

# ---------- SIMULATION CORE (Look-ahead Safe) ----------

def run_ltf_simulation_memory(ltf_data, start_index, entry, sl, tp, side, expiration_ts):
    is_open, real_entry_ts = False, 0
    for i in range(start_index, len(ltf_data)):
        row = ltf_data[i]
        ts, high, low = row['ts'], float(row['high']), float(row['low'])
        if not is_open:
            if ts > expiration_ts: return "EXPIRED", 0, ts
            if (side == "LONG" and low <= entry) or (side == "SHORT" and high >= entry):
                is_open, real_entry_ts = True, ts
                if (side == "LONG" and low <= sl) or (side == "SHORT" and high >= sl): return "LOSS", real_entry_ts, ts
        else:
            if (side == "LONG" and low <= sl) or (side == "SHORT" and high >= sl): return "LOSS", real_entry_ts, ts
            if (side == "LONG" and high >= tp) or (side == "SHORT" and low <= tp): return "WIN", real_entry_ts, ts
    return "EXPIRED", 0, expiration_ts

def detect_fvg_setup(rates, i, scale, allow_longs, allow_shorts):
    vision_limit = i - SWING_CONFIRMATION_LAG
    if vision_limit < 200: return None
    curr = rates[i]
    ema, htf = curr.get('ema_trend'), curr.get('htf_trend', 0)
    if not ema or pd.isna(ema): return None

    if allow_longs and curr['close'] > ema and htf == 1:
        sh_idx = next((k for k in range(vision_limit, vision_limit-60, -1) if rates[k]['is_swing_high']), -1)
        sl_idx = next((k for k in range(sh_idx-1, sh_idx-100, -1) if rates[k]['is_swing_low']), -1) if sh_idx != -1 else -1
        if sl_idx != -1:
            fib_p = rates[sh_idx]['high'] - ((rates[sh_idx]['high'] - rates[sl_idx]['low']) * FIB_RETREACEMENT)
            if fib_p > ema and curr['close'] > fib_p:
                return {"side": "LONG", "entry_price": qround(fib_p, scale), "sl_price": qround(rates[sl_idx]['low'], scale)}
    elif allow_shorts and curr['close'] < ema and htf == -1:
        sl_idx = next((k for k in range(vision_limit, vision_limit-60, -1) if rates[k]['is_swing_low']), -1)
        sh_idx = next((k for k in range(sl_idx-1, sl_idx-100, -1) if rates[k]['is_swing_high']), -1) if sl_idx != -1 else -1
        if sh_idx != -1:
            fib_p = rates[sl_idx]['low'] + ((rates[sh_idx]['high'] - rates[sl_idx]['low']) * FIB_RETREACEMENT)
            if fib_p < ema and curr['close'] < fib_p:
                return {"side": "SHORT", "entry_price": qround(fib_p, scale), "sl_price": qround(rates[sh_idx]['high'], scale)}
    return None

# ---------- EXECUTION AVEC OPTIMISATION RR ----------

def execute_backtest(engine, pair, rr_ratio, scale, start_ms, end_ms, risk_per_trade, allow_longs, allow_shorts):
    df_scan = fetch_htf_data_pandas_raw(engine, pair, SCAN_TF, start_ms, end_ms)
    if df_scan.empty: return []
    df_filter = fetch_trend_structure_data(engine, pair, TREND_FILTER_TF, start_ms, end_ms)
    df_scan = pd.merge_asof(df_scan.sort_values('time'), df_filter.sort_values('time'), on='time', direction='backward') if not df_filter.empty else df_scan.assign(htf_trend=0)
    
    rates = df_scan.to_dict('records')
    ltf_data = fetch_ltf_data_pandas(engine, pair, rates[0]['time'], rates[-1]['time'])
    ltf_ts = [r['ts'] for r in ltf_data]
    
    start_idx = next((idx for idx in range(200, len(rates)) if not start_ms or rates[idx]['time'] >= start_ms), len(rates))
    
    balance_r, wins, losses, trade_log, all_pnl = Decimal(0), 0, 0, [], []
    skip_until_ts, scan_dur = 0, rates[1]['time'] - rates[0]['time']

    for i in range(start_idx, len(rates)):
        if end_ms and rates[i]['time'] > end_ms: break
        if rates[i]['time'] < skip_until_ts: continue
        
        setup = detect_fvg_setup(rates, i, scale, allow_longs, allow_shorts)
        if setup:
            risk_dist = abs(setup["entry_price"] - setup["sl_price"])
            if risk_dist == 0: continue
            tp_p = qround(setup["entry_price"] + risk_dist * rr_ratio if setup["side"] == "LONG" else setup["entry_price"] - risk_dist * rr_ratio, scale)
            
            # Invalidation EMA 5m
            invalid = False
            for wait in range(1, MAX_WAIT_CANDLES + 1):
                c_idx = i + wait
                if c_idx >= len(rates): break
                c = rates[c_idx]
                if (setup["side"] == "LONG" and c['close'] < c['ema_trend']) or (setup["side"] == "SHORT" and c['close'] > c['ema_trend']):
                    invalid = True; skip_until_ts = c['time']; break
                if (setup["side"] == "LONG" and c['low'] <= float(setup["entry_price"])) or (setup["side"] == "SHORT" and c['high'] >= float(setup["entry_price"])): break
            
            if invalid: continue

            sim_start = rates[i]['time'] + scan_dur
            exp_ts = sim_start + (MAX_WAIT_CANDLES * scan_dur)
            ltf_idx = bisect.bisect_left(ltf_ts, sim_start)
            if ltf_idx >= len(ltf_data): continue

            res, entry_t, exit_t = run_ltf_simulation_memory(ltf_data, ltf_idx, float(setup["entry_price"]), float(setup["sl_price"]), float(tp_p), setup["side"], exp_ts)
            
            if res in ["WIN", "LOSS"]:
                pnl = rr_ratio if res == "WIN" else Decimal("-1.0")
                balance_r += pnl
                if res == "WIN": wins += 1
                else: losses += 1
                all_pnl.append(float(pnl))
                trade_log.append({"pair": pair, "entry_time": entry_t, "exit_time": exit_t, "side": setup["side"], "result": res, "pnl_r": pnl, "rr": rr_ratio})
            skip_until_ts = exit_t

    return trade_log

# ---------- MAIN AVEC BOUCLE RR ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--end-date", type=str, default=None)
    args = ap.parse_args()
    
    start_ms = parse_date_to_ms(args.start_date) if args.start_date else None
    end_ms = parse_date_to_ms(args.end_date, True) if args.end_date else None
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    
    best_overall_trades = {}

    print(f"--- OPTIMISATION RR PAR PAIRE ({RR_STEPS[0]} à {RR_STEPS[-1]}) ---")
    
    for p in pairs:
        scale = price_scale(p[:3], p[3:])
        best_rr_for_pair = RR_STEPS[0]
        best_expectancy = Decimal("-999")
        best_trades = []
        best_stats = {}

        for rr in RR_STEPS:
            trades = execute_backtest(engine, p, rr, scale, start_ms, end_ms, DEFAULT_RISK_PER_TRADE, True, True)
            
            total = len(trades)
            if total > 0:
                net_r = sum(t['pnl_r'] for t in trades)
                exp = net_r / total
                
                # Critère de sélection : Meilleure Expectancy (R moyen par trade)
                if exp > best_expectancy and total >= 5: # Minimum 5 trades pour être significatif
                    best_expectancy = exp
                    best_rr_for_pair = rr
                    best_trades = trades
                    
                    # Calcul SQN
                    pnls = [float(t['pnl_r']) for t in trades]
                    sqn = (math.sqrt(total) * statistics.mean(pnls) / statistics.stdev(pnls)) if total > 1 else 0
                    
                    best_stats = {
                        "pair": p, "trades": total, "rr": rr, "exp": exp, 
                        "wr": (len([t for t in trades if t['result'] == "WIN"]) / total) * 100,
                        "sqn": sqn, "net": net_r
                    }

        if best_trades:
            best_overall_trades[p] = best_trades
            GLOBAL_RESULTS.append(best_stats)
            print(f"[{p}] Best RR: {best_rr_for_pair} | Expectancy: {best_expectancy:.3f}R | Trades: {len(best_trades)}")
        else:
            print(f"[{p}] Aucun trade détecté.")

    # Affichage final
    if GLOBAL_RESULTS:
        df_res = pd.DataFrame(GLOBAL_RESULTS).sort_values("sqn", ascending=False)
        print("\n" + "="*100)
        print(f"RÉSULTATS FINAUX OPTIMISÉS (Filtré par SQN)")
        print("="*100)
        print(df_res.to_string(index=False))
        
        # Simulation Portefeuille
        total_pnl_r = sum(r['net'] for r in GLOBAL_RESULTS)
        print("\n" + "="*50)
        print(f"PERFORMANCE PORTEFEUILLE: {total_pnl_r:+.2f}R")
        print(f"BALANCE FINALE EST.: {INITIAL_BALANCE * (1 + total_pnl_r * DEFAULT_RISK_PER_TRADE):,.2f} USD")
        print("="*50)

if __name__ == "__main__":
    main()