#!/usr/bin/env python3
# postgres_fvg_backtester_GOLD_V32_AGGRESSIVE.py

import os
import re
import argparse
import bisect
import csv
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict

from prettytable import PrettyTable
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ---------- CONFIGURATION ----------
load_dotenv()
UTC = timezone.utc
SCAN_TF = "15m" 
INITIAL_BALANCE = Decimal("100000.00")
RISK_PER_TRADE = Decimal("0.001") # 0.1% par trade (pour tester le volume)
DATE_FORMAT = "%Y-%m-%d"

TF_TO_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000}
STEP_MS = TF_TO_MS.get(SCAN_TF, 900000)

class RobotMemory:
    def __init__(self):
        self.status = "EMPTY"
        self.order = None
        self.position = None

# ==============================================================================
#  MOTEUR AGRESSIF
# ==============================================================================

def run_backtest_aggressive(engine, pair: str, rr_long: Decimal, rr_short: Decimal, 
                            min_dist: Decimal, start_ms: int, end_ms: int, 
                            use_ema: bool, use_impulse: bool):
    
    pair_suffix = re.sub(r'[^a-z0-9]+', '_', pair.lower())
    table_scan = f"candles_mt5_{pair_suffix}_{SCAN_TF}"

    print(f"Chargement des données {SCAN_TF}...")
    # On ne charge pas ema_200 de la base, on va calculer une EMA plus rapide
    query_scan = f"SELECT ts as time, open, high, low, close FROM {table_scan} WHERE ts >= {start_ms - 86400000 * 5} AND ts <= {end_ms + 43200000} ORDER BY ts ASC"
    df_scan = pd.read_sql(query_scan, engine)
    if df_scan.empty: return []

    # CALCUL EMA 50 (Plus réactive)
    df_scan['ema_fast'] = df_scan['close'].ewm(span=50, adjust=False).mean()
    rates_scan = df_scan.to_dict('records')
    
    # CHARGEMENT LTF
    print("Chargement des données LTF (1m)...")
    table_ltf = f"candles_mt5_{pair_suffix}_1m"
    # Optimisation : on ne charge que ce qu'il faut
    query_ltf = f"SELECT ts, open, high, low, close FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {end_ms + 43200000} ORDER BY ts ASC"
    # Attention : charger toute la base 1m peut être lourd. Assure-toi que la plage de date est raisonnable.
    ltf_data = pd.read_sql(query_ltf, engine).to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data] 

    logs = []
    robot = RobotMemory()
    current_ltf_idx = 0

    print("Démarrage de la simulation...")
    # On commence après le buffer de calcul EMA
    start_index = next((i for i, r in enumerate(rates_scan) if r['time'] >= start_ms), 10)

    for i in range(start_index, len(rates_scan)):
        curr_bar = rates_scan[i]
        now_ts = curr_bar['time']
        
        # --- SYNCHRO LTF ---
        if current_ltf_idx < len(ltf_ts) and ltf_ts[current_ltf_idx] < now_ts:
             current_ltf_idx = bisect.bisect_left(ltf_ts, now_ts)
        next_bar_time = rates_scan[i+1]['time'] if i+1 < len(rates_scan) else now_ts + STEP_MS

        # --- PHASE 1: GESTION TRADES (LTF) ---
        while current_ltf_idx < len(ltf_data):
            candle_1m = ltf_data[current_ltf_idx]
            if candle_1m['ts'] >= next_bar_time: break
            ts_1m = candle_1m['ts']
            
            # OPEN
            if robot.status == "OPEN":
                pos = robot.position
                hit_sl = (pos['side'] == "LONG" and candle_1m['low'] <= pos['sl']) or \
                         (pos['side'] == "SHORT" and candle_1m['high'] >= pos['sl'])
                hit_tp = (pos['side'] == "LONG" and candle_1m['high'] >= pos['tp']) or \
                         (pos['side'] == "SHORT" and candle_1m['low'] <= pos['tp'])
                
                res = "LOSS" if hit_sl else ("WIN" if hit_tp else None)
                if res:
                    logs.append({
                        "setup_time": pos['setup_ts'],
                        "entry_time": pos['entry_ts'],
                        "exit_time": ts_1m,
                        "res": res, 
                        "pnl": pos['rr'] if res == "WIN" else Decimal("-1.0"), 
                        "side": pos['side'],
                        "entry_price": pos['entry'],
                        "sl": pos['sl'],
                        "tp": pos['tp']
                    })
                    robot.status = "EMPTY"
                    robot.position = None
            
            # PENDING
            if robot.status == "PENDING":
                order = robot.order
                if ts_1m > order['expiration']:
                    robot.status = "EMPTY"
                    robot.order = None
                else:
                    triggered = (order['side'] == "LONG" and candle_1m['low'] <= order['entry']) or \
                                (order['side'] == "SHORT" and candle_1m['high'] >= order['entry'])
                    if triggered:
                        robot.status = "OPEN"
                        robot.position = {
                            "side": order['side'], "entry": order['entry'], "sl": order['sl'], "tp": order['tp'],
                            "rr": order['rr'], "setup_ts": order['setup_ts'], "entry_ts": ts_1m
                        }
                        robot.order = None
            current_ltf_idx += 1

        # --- PHASE 2: SIGNAL ---
        
        # 1. Filtre EMA (Optionnel et Rapide)
        ema_bias = None
        if use_ema:
            ema_val = Decimal(str(curr_bar['ema_fast']))
            close_val = Decimal(str(curr_bar['close']))
            ema_bias = "LONG" if close_val > ema_val else "SHORT"

        # 2. Indicateurs
        is_green = rates_scan[i-1]['close'] > rates_scan[i-1]['open']
        is_red = rates_scan[i-1]['close'] < rates_scan[i-1]['open']
        
        # Filtre Impulse (Optionnel)
        valid_impulse = True
        if use_impulse:
            body_impulse = abs(rates_scan[i-2]['close'] - rates_scan[i-2]['open'])
            avg_body = sum(abs(rates_scan[j]['close'] - rates_scan[j]['open']) for j in range(i-7, i-2)) / 5
            if body_impulse <= avg_body: valid_impulse = False

        setup = None
        
        # SETUP LONG (i-1 Low > i-3 High) = FVG créée
        # Condition EMA : Soit on l'ignore (None), soit elle doit matcher
        if rates_scan[i-1]['low'] > rates_scan[i-3]['high'] and is_green and valid_impulse:
            if not use_ema or ema_bias == "LONG":
                e, s = Decimal(str(rates_scan[i-1]['low'])), Decimal(str(rates_scan[i-3]['low']))
                setup = {"side": "LONG", "entry": e, "sl": s, "tp": e + (e - s) * rr_long, "rr": rr_long}
            
        # SETUP SHORT (i-1 High < i-3 Low)
        elif rates_scan[i-1]['high'] < rates_scan[i-3]['low'] and is_red and valid_impulse:
            if not use_ema or ema_bias == "SHORT":
                e, s = Decimal(str(rates_scan[i-1]['high'])), Decimal(str(rates_scan[i-3]['high']))
                setup = {"side": "SHORT", "entry": e, "sl": s, "tp": e - (s - e) * rr_short, "rr": rr_short}

        # Placement ordre (Cancel & Replace pour être agressif sur le dernier signal frais)
        if setup and abs(setup['entry'] - setup['sl']) >= min_dist:
            if robot.status in ["EMPTY", "PENDING"]:
                robot.status = "PENDING"
                robot.order = {
                    "side": setup['side'], "entry": setup['entry'], "sl": setup['sl'], "tp": setup['tp'],
                    "rr": setup['rr'], "setup_ts": now_ts, "expiration": now_ts + 3600000 # 1h expiration
                }

    return logs

# ==============================================================================
#  AFFICHAGE
# ==============================================================================

def display_results(trades, args):
    if not trades: print("\n[!] Aucun trade détecté."); return
    
    trades.sort(key=lambda x: x['setup_time'])
    g_equity = INITIAL_BALANCE
    g_peak = INITIAL_BALANCE
    g_max_dd_pct = Decimal("0")
    
    stats = {"LONG": {"t":0,"w":0,"p":Decimal(0)}, "SHORT": {"t":0,"w":0,"p":Decimal(0)}}

    rows = []
    for t in trades:
        gain = (g_equity * RISK_PER_TRADE) * t['pnl']
        g_equity += gain
        
        s = t['side']
        stats[s]["t"] += 1; stats[s]["p"] += t['pnl']
        if t['res'] == "WIN": stats[s]["w"] += 1

        if g_equity > g_peak: g_peak = g_equity
        dd = ((g_peak - g_equity) / g_peak) * 100
        if dd > g_max_dd_pct: g_max_dd_pct = dd
        
        rows.append([
            datetime.fromtimestamp(t['setup_time']/1000, tz=UTC).strftime('%Y-%m-%d %H:%M'),
            t['side'], f"{t['entry_price']:.2f}", f"{t['sl']:.2f}", t['res'], f"{t['pnl']:+.2f}", f"{g_equity:,.0f}"
        ])

    print("\n" + "="*100)
    print(f"🔥 GOLD V32 AGGRESSIVE | EMA: {'ON (50)' if args.use_ema else 'OFF'} | IMPULSE: {'ON' if args.use_impulse else 'OFF'}")
    print("="*100)
    
    total = len(trades)
    wr = (Decimal(stats["LONG"]["w"]+stats["SHORT"]["w"])/total)*100
    
    gt = PrettyTable(["Métrique", "Valeur"])
    gt.align = "l"
    gt.add_row(["Capital Final", f"{g_equity:,.2f} $"])
    gt.add_row(["Gain Total (%)", f"{((g_equity-INITIAL_BALANCE)/INITIAL_BALANCE*100):+.2f} %"])
    gt.add_row(["Trades Totaux", total])
    gt.add_row(["Win Rate", f"{wr:.2f} %"])
    gt.add_row(["Max DD", f"{g_max_dd_pct:.2f} %"])
    print(gt)

    st = PrettyTable(["Side", "Trades", "WinRate", "PnL (R)", "Expectancy"])
    for s in ["LONG", "SHORT"]:
        d = stats[s]
        if d["t"] > 0:
            st.add_row([s, d["t"], f"{(d['w']/d['t']*100):.1f}%", f"{d['p']:+.1f}R", f"{(d['p']/d['t']):.2f}R"])
    print(st)
    
    print("\n📝 DERNIERS TRADES")
    jt = PrettyTable(["Date", "Side", "Entry", "SL", "Res", "PnL", "Equity"])
    for r in rows[-20:]: jt.add_row(r)
    print(jt)
    
    if args.csv:
        f = f"gold_aggressive_{datetime.now().strftime('%H%M%S')}.csv"
        with open(f, 'w') as o:
            w = csv.writer(o); w.writerow(["Date","Side","Entry","SL","Res","PnL","Equity"])
            w.writerows(rows)
        print(f"[+] CSV: {f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--rr-long", type=float, default=2.0)
    ap.add_argument("--rr-short", type=float, default=2.0)
    ap.add_argument("--min-dist", type=float, default=0.0)