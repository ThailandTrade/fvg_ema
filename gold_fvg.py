#!/usr/bin/env python3
# postgres_fvg_backtester_GOLD_V33_2_FRIDAY_EXIT.py

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

# ==============================================================================
#  🟢 CONFIGURATION UTILISATEUR
# ==============================================================================
load_dotenv()
UTC = timezone.utc

# 1. DIRECTIONS AUTORISÉES
ENABLE_LONG = True             
ENABLE_SHORT = True           

# 2. PARAMÈTRES DE RISQUE
RR_LONG = Decimal("3.0")       
RR_SHORT = Decimal("3.0")      
RISK_PER_TRADE = Decimal("0.001") 
INITIAL_BALANCE = Decimal("100000.00")

# 3. FILTRES
USE_EMA_FILTER = True          
USE_IMPULSE_FILTER = True      

# 4. SYSTÈME
SCAN_TF = "15m"                
MIN_DIST_SL = Decimal("0.0")   
DATE_FORMAT = "%Y-%m-%d"

TF_TO_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000}
STEP_MS = TF_TO_MS.get(SCAN_TF, 900000)

class RobotMemory:
    def __init__(self):
        self.status = "EMPTY"
        self.order = None
        self.position = None

# ==============================================================================
#  MOTEUR DE BACKTEST
# ==============================================================================

def run_backtest_v33(engine, pair: str, start_ms: int, end_ms: int):
    
    print("\n" + "="*50)
    print(f"🔧 CONFIG CHARGÉE :")
    print(f"   MODE : {'LONG & SHORT' if ENABLE_LONG and ENABLE_SHORT else 'RESTRICTED'}")
    print(f"   EXIT : VENDREDI 21H30 UTC (Close 21H45)")
    print("="*50 + "\n")

    pair_suffix = re.sub(r'[^a-z0-9]+', '_', pair.lower())
    table_scan = f"candles_mt5_{pair_suffix}_{SCAN_TF}"

    print(f"Chargement des données {SCAN_TF}...")
    query_scan = f"SELECT ts as time, open, high, low, close FROM {table_scan} WHERE ts >= {start_ms - 86400000 * 5} AND ts <= {end_ms + 43200000} ORDER BY ts ASC"
    df_scan = pd.read_sql(query_scan, engine)
    if df_scan.empty: return []

    df_scan['ema_fast'] = df_scan['close'].ewm(span=50, adjust=False).mean()
    rates_scan = df_scan.to_dict('records')
    
    print("Chargement des données LTF (1m)...")
    table_ltf = f"candles_mt5_{pair_suffix}_1m"
    query_ltf = f"SELECT ts, open, high, low, close FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {end_ms + 43200000} ORDER BY ts ASC"
    ltf_data = pd.read_sql(query_ltf, engine).to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data] 

    logs = []
    robot = RobotMemory()
    current_ltf_idx = 0

    print("🚀 Démarrage de la simulation...")
    start_index = next((i for i, r in enumerate(rates_scan) if r['time'] >= start_ms), 10)

    for i in range(start_index, len(rates_scan)):
        curr_bar = rates_scan[i]
        now_ts = curr_bar['time']
        
        # --- SYNCHRO ---
        if current_ltf_idx < len(ltf_ts) and ltf_ts[current_ltf_idx] < now_ts:
             current_ltf_idx = bisect.bisect_left(ltf_ts, now_ts)
        next_bar_time = rates_scan[i+1]['time'] if i+1 < len(rates_scan) else now_ts + STEP_MS

        # --- PHASE 1: GESTION INTRADAY (LTF) ---
        while current_ltf_idx < len(ltf_data):
            candle_1m = ltf_data[current_ltf_idx]
            if candle_1m['ts'] >= next_bar_time: break
            ts_1m = candle_1m['ts']
            
            if robot.status == "OPEN":
                pos = robot.position
                hit_sl = (pos['side'] == "LONG" and candle_1m['low'] <= pos['sl']) or \
                         (pos['side'] == "SHORT" and candle_1m['high'] >= pos['sl'])
                hit_tp = (pos['side'] == "LONG" and candle_1m['high'] >= pos['tp']) or \
                         (pos['side'] == "SHORT" and candle_1m['low'] <= pos['tp'])
                
                res = "LOSS" if hit_sl else ("WIN" if hit_tp else None)
                if res:
                    pnl_val = pos['rr'] if res == "WIN" else Decimal("-1.0")
                    logs.append({
                        "setup_time": pos['setup_ts'],
                        "entry_time": pos['entry_ts'],
                        "exit_time": ts_1m,
                        "res": res, 
                        "pnl": pnl_val,
                        "side": pos['side'],
                        "entry_price": pos['entry'],
                        "sl": pos['sl'],
                        "tp": pos['tp']
                    })
                    robot.status = "EMPTY"
                    robot.position = None
            
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

        # --- PHASE 1.5 : FORÇAGE CLÔTURE VENDREDI ---
        # On vérifie l'heure de la bougie qui vient de se terminer (ou cours)
        curr_dt = datetime.fromtimestamp(now_ts / 1000, tz=UTC)
        
        # Vendredi (4) à 21h30
        if curr_dt.weekday() == 4 and curr_dt.hour == 21 and curr_dt.minute == 30:
            
            # 1. On annule les ordres en attente
            if robot.status == "PENDING":
                robot.status = "EMPTY"
                robot.order = None
            
            # 2. On ferme les positions ouvertes au prix de CLÔTURE de la bougie 21h30
            elif robot.status == "OPEN":
                pos = robot.position
                close_price = Decimal(str(curr_bar['close']))
                risk_dist = abs(pos['entry'] - pos['sl'])
                
                # Calcul du PnL réel en R
                if pos['side'] == "LONG":
                    real_pnl = (close_price - pos['entry']) / risk_dist
                else:
                    real_pnl = (pos['entry'] - close_price) / risk_dist
                
                logs.append({
                    "setup_time": pos['setup_ts'],
                    "entry_time": pos['entry_ts'],
                    "exit_time": next_bar_time, # Clôture de la bougie
                    "res": "FRIDAY", 
                    "pnl": real_pnl,
                    "side": pos['side'],
                    "entry_price": pos['entry'],
                    "sl": pos['sl'],
                    "tp": pos['tp']
                })
                robot.status = "EMPTY"
                robot.position = None
            
            # On passe direct à la suite sans chercher de nouveau setup sur cette bougie
            continue

        # --- PHASE 2: SIGNAL ---
        ema_bias = None
        if USE_EMA_FILTER:
            ema_val = Decimal(str(curr_bar['ema_fast']))
            close_val = Decimal(str(curr_bar['close']))
            ema_bias = "LONG" if close_val > ema_val else "SHORT"

        is_green = rates_scan[i-1]['close'] > rates_scan[i-1]['open']
        is_red = rates_scan[i-1]['close'] < rates_scan[i-1]['open']
        
        valid_impulse = True
        if USE_IMPULSE_FILTER:
            body_impulse = abs(rates_scan[i-2]['close'] - rates_scan[i-2]['open'])
            avg_body = sum(abs(rates_scan[j]['close'] - rates_scan[j]['open']) for j in range(i-7, i-2)) / 5
            if body_impulse <= avg_body: valid_impulse = False

        setup = None
        
        if ENABLE_LONG and rates_scan[i-1]['low'] > rates_scan[i-3]['high'] and is_green and valid_impulse:
            if not USE_EMA_FILTER or ema_bias == "LONG":
                e, s = Decimal(str(rates_scan[i-1]['low'])), Decimal(str(rates_scan[i-3]['low']))
                tp_price = e + (e - s) * RR_LONG
                setup = {"side": "LONG", "entry": e, "sl": s, "tp": tp_price, "rr": RR_LONG}
            
        elif ENABLE_SHORT and rates_scan[i-1]['high'] < rates_scan[i-3]['low'] and is_red and valid_impulse:
            if not USE_EMA_FILTER or ema_bias == "SHORT":
                e, s = Decimal(str(rates_scan[i-1]['high'])), Decimal(str(rates_scan[i-3]['high']))
                tp_price = e - (s - e) * RR_SHORT
                setup = {"side": "SHORT", "entry": e, "sl": s, "tp": tp_price, "rr": RR_SHORT}

        if setup and abs(setup['entry'] - setup['sl']) >= MIN_DIST_SL:
            if robot.status in ["EMPTY", "PENDING"]:
                robot.status = "PENDING"
                robot.order = {
                    "side": setup['side'], "entry": setup['entry'], "sl": setup['sl'], "tp": setup['tp'],
                    "rr": setup['rr'], "setup_ts": now_ts, "expiration": now_ts + 3600000 
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
            t['side'], 
            f"{t['entry_price']:.2f}", 
            f"{t['sl']:.2f}", 
            f"{t['tp']:.2f}", 
            t['res'], 
            f"{t['pnl']:+.2f}", 
            f"{g_equity:,.0f}"
        ])

    print("\n" + "="*100)
    print(f"🔥 GOLD V33.2 | FRIDAY EXIT ON | RR L:{RR_LONG} S:{RR_SHORT}")
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
    
    print("\n📝 JOURNAL (30 DERNIERS TRADES)")
    jt = PrettyTable(["Date", "Side", "Entry", "SL", "TP", "Res", "PnL", "Equity"])
    for r in rows[-30:]: jt.add_row(r)
    print(jt)
    
    if args.csv:
        f = f"gold_v33_weekend_{datetime.now().strftime('%H%M%S')}.csv"
        with open(f, 'w', newline='') as o:
            w = csv.writer(o); w.writerow(["Date","Side","Entry","SL","TP","Res","PnL","Equity"])
            w.writerows(rows)
        print(f"[+] CSV: {f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    engine = create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")
    s_ms = int(datetime.strptime(args.start_date, DATE_FORMAT).replace(tzinfo=UTC).timestamp() * 1000)
    e_ms = int(datetime.now(tz=UTC).timestamp() * 1000)

    trades = run_backtest_v33(engine, "XAUUSD.c", s_ms, e_ms)
    display_results(trades, args)