#!/usr/bin/env python3
# postgres_fvg_backtester_SEQUENTIAL_PRETTY.py

import os
import re
import csv
import sys
import argparse
import bisect 
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List, Any

# NOUVEAU : PrettyTable pour l'affichage
from prettytable import PrettyTable

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG ----------
DEFAULT_RR = Decimal("3.0")
MAX_WAIT_CANDLES = 72
SCAN_TF = "5m"            
EXECUTION_TF_SUFFIX = "1m" 
DEFAULT_RISK_PER_TRADE = Decimal("0.001")
INITIAL_BALANCE = Decimal("100000.00") 
FIB_RETREACEMENT = 0.62
SWING_CONFIRMATION_LAG = 5 

# ---------- UTILS ----------

def price_scale(pair: str) -> int:
    return 3 if any(x in pair for x in ["JPY", "XAU", "XAUUSD", "XAG"]) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime('%m-%d %H:%M')

def parse_pairs_with_rr(path: str) -> List[Tuple[str, Optional[Decimal]]]:
    out = []
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [x.strip().lower() for x in reader.fieldnames]
        for r in reader:
            p = r.get("pair") or r.get("asset")
            rr_val = r.get("rr") or r.get("risk_reward")
            if p: out.append((p.strip(), Decimal(rr_val) if rr_val else None))
    return out

# ---------- MOTEUR DE DÉTECTION SÉQUENTIEL (INCHANGÉ) ----------

def get_swings_realtime(df: pd.DataFrame):
    l = SWING_CONFIRMATION_LAG
    window = 2 * l + 1
    df['is_swing_high'] = df['high'].rolling(window=window, center=False).apply(lambda x: 1 if x.argmax() == l else 0, raw=True)
    df['is_swing_low'] = df['low'].rolling(window=window, center=False).apply(lambda x: 1 if x.argmin() == l else 0, raw=True)
    return df

def run_backtest_sequential(engine, pair: str, rr: Decimal, start_ms: int, end_ms: int):
    # Chargement Data 5m
    table = f"candles_mt5_{re.sub(r'[^a-z0-9]+', '_', pair.lower())}_{SCAN_TF}"
    query = f"SELECT ts as time, high, low, close, ema_200 FROM {table} WHERE ts >= {start_ms - 864000000} AND ts <= {end_ms} ORDER BY ts ASC"
    df = pd.read_sql(query, engine)
    if df.empty: return []

    df = get_swings_realtime(df)
    rates = df.to_dict('records')
    scale = price_scale(pair)

    # Chargement Data 1m
    table_ltf = f"candles_mt5_{re.sub(r'[^a-z0-9]+', '_', pair.lower())}_1m"
    query_ltf = f"SELECT ts, high, low, close FROM {table_ltf} WHERE ts >= {rates[0]['time']} AND ts <= {end_ms + 43200000} ORDER BY ts ASC"
    ltf_data = pd.read_sql(query_ltf, engine).to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data]

    logs = []
    skip_until_ts = 0 
    
    for i in range(len(rates)):
        current_t = rates[i]['time']
        start_exec_t = current_t + 300000 

        if start_exec_t < skip_until_ts: continue 
        if current_t < start_ms: continue
        
        ema = rates[i]['ema_200']
        if not ema or pd.isna(ema): continue

        setup = None
        
        # LONG
        if rates[i]['close'] > ema:
            sh_idx = -1
            for k in range(i, i - 60, -1):
                if rates[k]['is_swing_high'] == 1:
                    sh_idx = k - SWING_CONFIRMATION_LAG 
                    break
            if sh_idx != -1:
                sl_idx = -1
                for k in range(sh_idx - 1, sh_idx - 100, -1):
                    if rates[k]['is_swing_low'] == 1: 
                        sl_idx = k - SWING_CONFIRMATION_LAG
                        break
                if sl_idx != -1:
                    dist = rates[sh_idx]['high'] - rates[sl_idx]['low']
                    fib = rates[sh_idx]['high'] - (dist * FIB_RETREACEMENT)
                    if fib > ema and rates[i]['close'] > fib:
                        setup = {"side": "LONG", "entry": qround(fib, scale), "sl": qround(rates[sl_idx]['low'], scale)}

        # SHORT
        if not setup and rates[i]['close'] < ema:
            sl_idx = -1
            for k in range(i, i - 60, -1):
                if rates[k]['is_swing_low'] == 1:
                    sl_idx = k - SWING_CONFIRMATION_LAG
                    break
            if sl_idx != -1:
                sh_idx = -1
                for k in range(sl_idx - 1, sl_idx - 100, -1):
                    if rates[k]['is_swing_high'] == 1:
                        sh_idx = k - SWING_CONFIRMATION_LAG
                        break
                if sh_idx != -1:
                    dist = rates[sh_idx]['high'] - rates[sl_idx]['low']
                    fib = rates[sl_idx]['low'] + (dist * FIB_RETREACEMENT)
                    if fib < ema and rates[i]['close'] < fib:
                        setup = {"side": "SHORT", "entry": qround(fib, scale), "sl": qround(rates[sh_idx]['high'], scale)}

        # SIMULATION
        if setup:
            risk = abs(setup['entry'] - setup['sl'])
            if risk == 0: continue
            tp = qround(setup['entry'] + (risk * rr) if setup['side'] == "LONG" else setup['entry'] - (risk * rr), scale)
            
            exp_ts = current_t + (MAX_WAIT_CANDLES * 300000)
            idx_ltf = bisect.bisect_left(ltf_ts, start_exec_t)
            
            is_open = False; entry_t = 0
            
            for k in range(idx_ltf, len(ltf_data)):
                r_ltf = ltf_data[k]
                if r_ltf['ts'] > exp_ts: break
                
                if not is_open:
                    if (setup['side'] == "LONG" and r_ltf['low'] <= setup['entry']) or \
                       (setup['side'] == "SHORT" and r_ltf['high'] >= setup['entry']):
                        is_open = True; entry_t = r_ltf['ts']
                
                if is_open:
                    if (setup['side'] == "LONG" and r_ltf['low'] <= setup['sl']) or \
                       (setup['side'] == "SHORT" and r_ltf['high'] >= setup['sl']):
                        logs.append({"pair": pair, "setup_t": current_t, "entry_t": entry_t, "exit_t": r_ltf['ts'], "res": "LOSS", "pnl": Decimal("-1.0"), "side": setup['side'], "entry": setup['entry'], "sl": setup['sl'], "tp": tp})
                        skip_until_ts = r_ltf['ts']
                        break
                    
                    if (setup['side'] == "LONG" and r_ltf['high'] >= tp) or \
                       (setup['side'] == "SHORT" and r_ltf['low'] <= tp):
                        logs.append({"pair": pair, "setup_t": current_t, "entry_t": entry_t, "exit_t": r_ltf['ts'], "res": "WIN", "pnl": rr, "side": setup['side'], "entry": setup['entry'], "sl": setup['sl'], "tp": tp})
                        skip_until_ts = r_ltf['ts'] 
                        break
    return logs

# ---------- DISPLAY (PRETTYTABLE) ----------

def display_results(all_trades):
    if not all_trades:
        print("\n[!] Aucun trade trouvé.")
        return

    all_trades.sort(key=lambda x: x['setup_t'])
    
    # 1. TABLEAU DÉTAILLÉ
    table = PrettyTable()
    table.field_names = ["Paire", "Setup (UTC)", "Exit (UTC)", "Side", "Entry", "SL", "TP", "Res", "PnL"]
    
    # Alignement
    table.align["Paire"] = "l"
    table.align["Side"] = "c"
    table.align["Res"] = "c"
    table.align["PnL"] = "r"
    table.align["Entry"] = "r"
    table.align["SL"] = "r"
    table.align["TP"] = "r"

    total_pnl = Decimal("0.0")
    wins = 0

    for t in all_trades:
        total_pnl += t['pnl']
        if t['res'] == "WIN": wins += 1
        
        # Formatage dynamique selon la paire
        scale = price_scale(t['pair'])
        entry_str = f"{float(t['entry']):.{scale}f}"
        sl_str = f"{float(t['sl']):.{scale}f}"
        tp_str = f"{float(t['tp']):.{scale}f}"
        pnl_str = f"{float(t['pnl']):+.2f}R"

        table.add_row([
            t['pair'],
            format_ts(t['setup_t']),
            format_ts(t['exit_t']),
            t['side'],
            entry_str,
            sl_str,
            tp_str,
            t['res'],
            pnl_str
        ])

    print("\n" + str(table))

    # 2. TABLEAU RÉCAPITULATIF
    summary = PrettyTable()
    summary.field_names = ["Métrique", "Valeur"]
    summary.align = "l"
    
    win_rate = (wins / len(all_trades)) * 100
    
    summary.add_row(["Total Trades", str(len(all_trades))])
    summary.add_row(["Win Rate", f"{win_rate:.2f}%"])
    summary.add_row(["Total PnL", f"{total_pnl:+.2f} R"])
    
    # Estimation gain monétaire (sur 100k capital, 0.1% risque)
    monetary_gain = INITIAL_BALANCE * DEFAULT_RISK_PER_TRADE * total_pnl
    summary.add_row(["Gain Est. ($)", f"{monetary_gain:+.2f} $"])

    print("\n" + str(summary))

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", required=True)
    ap.add_argument("--start-date", required=True)
    args = ap.parse_args()

    engine = create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")
    pairs = parse_pairs_with_rr(args.pairs_file)
    s_ms = int(datetime.strptime(args.start_date, DATE_FORMAT).replace(tzinfo=UTC).timestamp() * 1000)
    e_ms = int(datetime.now().timestamp() * 1000)

    global_trades = []
    print(f"\n🚀 Lancement Backtest Séquentiel (Start: {args.start_date})...")
    
    for p_name, p_rr in pairs:
        print(f"  > Scanning {p_name}...")
        res = run_backtest_sequential(engine, p_name, p_rr or DEFAULT_RR, s_ms, e_ms)
        global_trades.extend(res)

    display_results(global_trades)

if __name__ == "__main__":
    main()