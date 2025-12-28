#!/usr/bin/env python3
# postgres_liquidity_raid_backtester.py

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

# ---------- CONFIG DE TRADING ----------
DEFAULT_RR = Decimal("2.0")       # Raid strategy offre souvent de bons RR
MAX_WAIT_CANDLES = 6              # On laisse un peu de temps au FVG de se former après le raid
SCAN_TF = "30m"                    # 1h est idéal pour voir les raids de PDH/PDL
EXECUTION_TF_SUFFIX = "1m" 
DEFAULT_RISK_PER_TRADE = Decimal("0.003") 
DEFAULT_FEES_PCT = Decimal("0.10") 

# --- PARAMETRES DU SUMMARY ---
INITIAL_BALANCE = Decimal("50000.00") 
SHOW_ALL_TRADES = True 

# ---------- CONSTANTES POUR STDEV (FVG) ----------
STDEV_PERIOD = 200 
DEFAULT_STDEV_THRESHOLD = 0.5
DEFAULT_STDEV_MAX = 1.0

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


# ---------- CALCUL PDH / PDL (NOUVEAU) ----------

def calculate_daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le Previous Day High (PDH) et Previous Day Low (PDL)
    et les injecte dans le DataFrame intraday.
    """
    # 1. Copie pour ne pas modifier l'index original
    df_calc = df.copy()
    df_calc['datetime'] = pd.to_datetime(df_calc['time'], unit='ms', utc=True)
    
    # 2. Resampling en Daily pour trouver High/Low du jour
    # '1D' peut créer des décalages selon l'heure de début, on assure minuit UTC
    df_daily = df_calc.resample('D', on='datetime').agg({
        'high': 'max',
        'low': 'min'
    })
    
    # 3. Shift(1) : Le PDH d'aujourd'hui est le High d'hier
    df_daily['pdh'] = df_daily['high'].shift(1)
    df_daily['pdl'] = df_daily['low'].shift(1)
    
    # 4. Préparation pour le merge (on ne garde que la date sans l'heure)
    df_daily['date_key'] = df_daily.index.date
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['date_key'] = df['datetime'].dt.date
    
    # 5. Fusion : Chaque bougie 1h récupère le PDH/PDL de sa journée
    df_merged = pd.merge(df, df_daily[['date_key', 'pdh', 'pdl']], on='date_key', how='left')
    
    # Nettoyage
    return df_merged.drop(columns=['datetime', 'date_key'])


# --- FETCH DATA ---

def fetch_htf_data_pandas(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"
    query = f"SELECT ts as time, open, high, low, close FROM {table_name}" # Plus besoin d'EMA
    
    # Buffer pour calculer le PDH de la veille
    safe_buffer_ms = timedelta(days=10).total_seconds() * 1000
    cond = []
    if start_ms: cond.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms: cond.append(f"ts <= {end_ms}")
    if cond: query += " WHERE " + " AND ".join(cond)
    query += " ORDER BY ts ASC"

    try:
        df = pd.read_sql(query, engine)
        if df.empty: return None

        # 1. Calcul FVG StDev (Vectorisé)
        gap_series = (df['low'] - df['high'].shift(2)).abs()
        df['stdev_200'] = gap_series.rolling(window=STDEV_PERIOD).std(ddof=1).fillna(0.0)
        
        # 2. Calcul PDH / PDL
        df = calculate_daily_levels(df)
        
        # On enlève les lignes où PDH/PDL sont NaN (début d'historique)
        df = df.dropna(subset=['pdh', 'pdl'])

        return df.to_dict('records')
    except Exception as e:
        print(f"[ERR] Fetch HTF: {e}")
        return None

def fetch_ltf_data_pandas(engine, pair: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    buffer_end = end_ms + (MAX_WAIT_CANDLES * 60 * 60 * 1000 * 2) 
    query = f"SELECT ts, high, low FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {buffer_end} ORDER BY ts ASC"
    try:
        df = pd.read_sql(query, engine)
        return [] if df.empty else df.to_dict('records')
    except Exception: return []

# --- SIMULATION MEMOIRE (INCHANGÉE) ---

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


# --- LOGIQUE LIQUIDITY RAID + FVG ---

def check_fvg_volatility(rates, i, threshold):
    # Même logique FVG que V3
    if i < 2: return False, False, 0.0, 0.0
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

def detect_raid_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_min: float, stdev_max: float) -> Optional[Dict[str, Any]]:
    """
    Stratégie: 
    1. Le prix a 'raidé' (méché) le PDH ou PDL.
    2. Un FVG se forme dans le sens inverse pour confirmer le rejet.
    """
    row = rates[i]
    pdh = row['pdh']
    pdl = row['pdl']
    
    if pdh is None or pdl is None or np.isnan(pdh) or np.isnan(pdl): return None

    # On vérifie si un FVG s'est formé sur CETTE bougie
    is_bull_fvg, is_bear_fvg, score, gap = check_fvg_volatility(rates, i, stdev_min)
    
    if score > stdev_max: return None # FVG trop gros (news ?)
    if not is_bull_fvg and not is_bear_fvg: return None

    entry_price = Decimal(0); sl_price = Decimal(0); side = ""

    # --- SCÉNARIO SHORT (RAID PDH) ---
    # Condition : FVG Baissier + Le High de la bougie actuelle ou précédente a cassé le PDH
    if is_bear_fvg:
        # On regarde si la bougie qui a créé le FVG (i) ou celle d'avant (i-1) ou (i-2) a pris le PDH
        # Le raid doit être récent.
        raid_occured = (rates[i]['high'] > pdh) or (rates[i-1]['high'] > pdh) or (rates[i-2]['high'] > pdh)
        
        if raid_occured:
            side = "SHORT"
            # Entrée FVG Classique (50%)
            fvg_top = Decimal(str(rates[i-2]["low"]))
            fvg_bot = Decimal(str(rates[i]["high"]))
            entry_price = (fvg_top + fvg_bot) / Decimal("2.0")
            
            # SL au dessus du plus haut du Raid (pour être safe)
            # On prend le max des 3 dernières bougies pour trouver le sommet de la mèche qui a raidé
            raid_high = max(rates[i]['high'], rates[i-1]['high'], rates[i-2]['high'])
            sl_price = Decimal(str(raid_high))
            
            if sl_price <= entry_price: return None # Incohérent

    # --- SCÉNARIO LONG (RAID PDL) ---
    # Condition : FVG Haussier + Le Low récent a cassé le PDL
    elif is_bull_fvg:
        raid_occured = (rates[i]['low'] < pdl) or (rates[i-1]['low'] < pdl) or (rates[i-2]['low'] < pdl)
        
        if raid_occured:
            side = "LONG"
            fvg_top = Decimal(str(rates[i]["low"]))
            fvg_bot = Decimal(str(rates[i-2]["high"]))
            entry_price = (fvg_top + fvg_bot) / Decimal("2.0")
            
            raid_low = min(rates[i]['low'], rates[i-1]['low'], rates[i-2]['low'])
            sl_price = Decimal(str(raid_low))
            
            if sl_price >= entry_price: return None

    if not side: return None

    return {
        "side": side,
        "entry_price": qround(entry_price, scale),
        "sl_price": qround(sl_price, scale),
        "pdh": pdh, "pdl": pdl # Pour debug/info
    }


# ---------- EXECUTION ----------

def execute_backtest(engine, pair, rr, scale, stdev_min, start_ms, end_ms, risk, stdev_max, fees):
    rates = fetch_htf_data_pandas(engine, pair, SCAN_TF, start_ms, end_ms)
    if not rates or len(rates) < 200: return []
    
    ltf_data = fetch_ltf_data_pandas(engine, pair, rates[0]['time'], rates[-1]['time'])
    ltf_ts = [r['ts'] for r in ltf_data]
    
    start_index = 0
    seed = STDEV_PERIOD + 2
    for idx in range(seed, len(rates)):
        if start_ms is None or rates[idx]['time'] >= start_ms:
            start_index = idx; break
    else: return []

    balance_r, total, wins, losses = Decimal(0), 0, 0, 0
    trade_log, all_pnl_r = [], []
    g_profit, g_loss = Decimal(0), Decimal(0)
    
    scan_ms = rates[1]['time'] - rates[0]['time'] if len(rates) > 1 else 3600000
    skip_until = 0

    for i in range(start_index, len(rates)):
        if end_ms and rates[i]['time'] > end_ms: break
        if rates[i]['time'] < skip_until: continue
        
        setup = detect_raid_setup(rates, i, scale, stdev_min, stdev_max)
        if setup:
            risk_amt = abs(setup["entry_price"] - setup["sl_price"])
            if setup["side"] == "LONG": tp_price = setup["entry_price"] + (risk_amt * rr)
            else: tp_price = setup["entry_price"] - (risk_amt * rr)
            
            sim_start = rates[i]['time'] + scan_ms
            l_idx = bisect.bisect_left(ltf_ts, sim_start)
            if l_idx >= len(ltf_data): continue
            
            # On donne un peu plus de temps pour ce genre de setup (reversal)
            expire = sim_start + (MAX_WAIT_CANDLES * scan_ms)
            
            res, exit_t = run_ltf_simulation_memory(ltf_data, l_idx, float(setup["entry_price"]), float(setup["sl_price"]), float(tp_price), setup["side"], expire)
            
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
                    "tp_price": qround(tp_price, scale), "result": res, "pnl_r": pnl
                })
                skip_until = exit_t

    if total > 0:
        pf = g_profit / g_loss if g_loss > 0 else Decimal("99.9")
        sqn = (math.sqrt(total) * (statistics.mean(all_pnl_r) / statistics.stdev(all_pnl_r))) if total > 1 and statistics.stdev(all_pnl_r) > 0 else 0
        GLOBAL_RESULTS.append({
            "pair": pair, "total_trades": total, "wins": wins, "losses": losses,
            "expectancy_r": balance_r/total, "win_rate": (wins/total*100),
            "profit_factor": pf, "sqn": sqn, "max_drawdown_r": 0,
            "start_ts": rates[start_index]['time'], "end_ts": rates[-1]['time']
        })
    return trade_log

# ---------- AFFICHAGE ----------

def display_summary(rr, results):
    results.sort(key=lambda x: x['sqn'], reverse=True)
    print(f"\nSUMMARY LIQUIDITY RAID (PDH/PDL + FVG) (RR: {rr}R)")
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