#!/usr/bin/env python3
# postgres_tokyo_orb_breakout.py

import os
import re
import argparse
import bisect
import csv
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict

from prettytable import PrettyTable
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ---------- CONFIGURATION ----------
load_dotenv()
UTC = timezone.utc
INITIAL_BALANCE = Decimal("100000.00")
RISK_PER_TRADE = Decimal("0.01") # 1% par trade (ajustable)
DATE_FORMAT = "%Y-%m-%d"

# Conversion TF pour la simulation interne
TF_TO_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000, "1h": 3600000}

class RobotMemory:
    def __init__(self):
        self.status = "EMPTY"
        self.position = None
        self.last_trade_day = None # Pour éviter de trader 2 fois le même jour

# ==============================================================================
#  MOTEUR TOKYO ORB
# ==============================================================================

def get_daily_ranges(engine, pair_suffix, start_ms, end_ms):
    """
    Récupère les bougies 1H de 00:00 UTC pour définir les ranges.
    Retourne un dictionnaire : { 'YYYY-MM-DD': {'high': ..., 'low': ...} }
    """
    print("Construction des ranges Tokyo (00:00 UTC)...")
    table_1h = f"candles_mt5_{pair_suffix}_1h"
    # On prend un buffer avant pour être sûr d'avoir le range du jour de démarrage
    query = f"""
    SELECT ts, high, low 
    FROM {table_1h} 
    WHERE ts >= {start_ms - 86400000} AND ts <= {end_ms}
    ORDER BY ts ASC
    """
    df = pd.read_sql(query, engine)
    ranges = {}
    for r in df.to_dict('records'):
        # On convertit le timestamp en Date UTC
        dt = datetime.fromtimestamp(r['ts'] / 1000, tz=UTC)
        # On ne garde que la bougie de 00:00
        if dt.hour == 0:
            date_key = dt.strftime('%Y-%m-%d')
            ranges[date_key] = {
                'high': Decimal(str(r['high'])), 
                'low': Decimal(str(r['low'])),
                'mid': (Decimal(str(r['high'])) + Decimal(str(r['low']))) / 2
            }
    return ranges

def run_backtest_orb(engine, pair: str, scan_tf: str, rr: Decimal, start_ms: int, end_ms: int):
    
    pair_suffix = re.sub(r'[^a-z0-9]+', '_', pair.lower())
    step_ms = TF_TO_MS.get(scan_tf, 900000)

    # 1. RÉCUPÉRATION DES RANGES (00:00 UTC)
    daily_ranges = get_daily_ranges(engine, pair_suffix, start_ms, end_ms)
    
    # 2. CHARGEMENT DES DONNÉES DE SCAN (ex: 15m)
    table_scan = f"candles_mt5_{pair_suffix}_{scan_tf}"
    print(f"Chargement des données de scan {scan_tf}...")
    query_scan = f"SELECT ts as time, open, high, low, close FROM {table_scan} WHERE ts >= {start_ms} AND ts <= {end_ms} ORDER BY ts ASC"
    df_scan = pd.read_sql(query_scan, engine)
    if df_scan.empty: return []
    rates_scan = df_scan.to_dict('records')

    # 3. CHARGEMENT LTF (1m) POUR LA PRÉCISION DU SL/TP
    print("Chargement des données LTF (1m) pour simulation précise...")
    table_ltf = f"candles_mt5_{pair_suffix}_1m"
    query_ltf = f"SELECT ts, open, high, low, close FROM {table_ltf} WHERE ts >= {rates_scan[0]['time']} AND ts <= {end_ms + 43200000} ORDER BY ts ASC"
    ltf_data = pd.read_sql(query_ltf, engine).to_dict('records')
    ltf_ts = [r['ts'] for r in ltf_data] 

    logs = []
    robot = RobotMemory()
    current_ltf_idx = 0

    # --- BOUCLE PRINCIPALE ---
    for i in range(len(rates_scan)):
        curr_bar = rates_scan[i]
        now_ts = curr_bar['time']
        
        # Gestion date et heure
        curr_dt = datetime.fromtimestamp(now_ts / 1000, tz=UTC)
        date_key = curr_dt.strftime('%Y-%m-%d')
        
        # Synchronisation index LTF
        if current_ltf_idx < len(ltf_ts) and ltf_ts[current_ltf_idx] < now_ts:
             current_ltf_idx = bisect.bisect_left(ltf_ts, now_ts)
        
        # Fin de la bougie de scan actuelle (pour simulation interne)
        next_bar_time = now_ts + step_ms

        # ==============================================================================
        # PHASE 1 : GESTION DES POSITIONS OUVERTES (Simulation bougie par bougie 1m)
        # ==============================================================================
        if robot.position is not None:
            while current_ltf_idx < len(ltf_data):
                candle_1m = ltf_data[current_ltf_idx]
                if candle_1m['ts'] >= next_bar_time: break # On sort si on dépasse la bougie 15m
                
                pos = robot.position
                ts_1m = candle_1m['ts']

                # Vérification SL / TP
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
                        "pnl": rr if res == "WIN" else Decimal("-1.0"), 
                        "side": pos['side'],
                        "entry_price": pos['entry'],
                        "sl": pos['sl'],
                        "tp": pos['tp'],
                        "range_high": pos['range_high'],
                        "range_low": pos['range_low']
                    })
                    robot.position = None
                    robot.status = "EMPTY"
                    break # Position fermée, on sort de la boucle LTF
                
                current_ltf_idx += 1
        else:
            # Si pas de position, on avance juste l'index
            current_ltf_idx = bisect.bisect_left(ltf_ts, next_bar_time)

        # ==============================================================================
        # PHASE 2 : DÉCISION (PRISE DE POSITION)
        # ==============================================================================
        # Conditions : 
        # 1. Pas de position en cours
        # 2. Pas encore tradé ce jour là (Une seule cassure par jour)
        # 3. Le range du jour existe
        # 4. Il est plus de 01:00 (le range 00h-01h doit être terminé)
        
        if robot.status == "EMPTY" and robot.last_trade_day != date_key:
            if date_key in daily_ranges:
                rng = daily_ranges[date_key]
                
                # On ne trade pas PENDANT la formation du range (avant 01:00 UTC)
                if curr_dt.hour >= 1:
                    close_price = Decimal(str(curr_bar['close']))
                    
                    setup = None
                    
                    # BREAKOUT HAUSSIER
                    if close_price > rng['high']:
                        setup = "LONG"
                        entry = close_price
                        sl = rng['mid']
                        dist = entry - sl
                        tp = entry + (dist * rr)
                    
                    # BREAKOUT BAISSIER
                    elif close_price < rng['low']:
                        setup = "SHORT"
                        entry = close_price
                        sl = rng['mid']
                        dist = sl - entry
                        tp = entry - (dist * rr)
                    
                    if setup:
                        # Entrée immédiate (Market au Close de la bougie signal)
                        robot.status = "OPEN"
                        robot.last_trade_day = date_key # On verrouille pour la journée
                        robot.position = {
                            "side": setup,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "setup_ts": now_ts,
                            "entry_ts": next_bar_time, # On considère l'entrée à l'ouverture bougie suivante (ou fin de celle-ci)
                            "range_high": rng['high'],
                            "range_low": rng['low']
                        }

    return logs

# ==============================================================================
#  AFFICHAGE
# ==============================================================================

def display_results(trades, args):
    if not trades:
        print("\n[!] Aucun trade détecté."); return
    
    trades.sort(key=lambda x: x['setup_time'])
    g_equity = INITIAL_BALANCE
    g_peak = INITIAL_BALANCE
    g_max_dd_pct = Decimal("0")
    
    stats = {
        "LONG": {"total": 0, "win": 0, "pnl": Decimal("0")},
        "SHORT": {"total": 0, "win": 0, "pnl": Decimal("0")}
    }

    journal_rows = []
    for t in trades:
        risk_amount = g_equity * RISK_PER_TRADE
        gain_usd = risk_amount * t['pnl']
        g_equity += gain_usd
        
        s = t['side']
        stats[s]["total"] += 1
        stats[s]["pnl"] += t['pnl']
        if t['res'] == "WIN": stats[s]["win"] += 1

        if g_equity > g_peak: g_peak = g_equity
        dd_pct = ((g_peak - g_equity) / g_peak) * 100
        if dd_pct > g_max_dd_pct: g_max_dd_pct = dd_pct
        
        d_setup = datetime.fromtimestamp(t['setup_time']/1000, tz=UTC).strftime('%Y-%m-%d %H:%M')
        journal_rows.append([
            d_setup, t['side'], 
            f"{t['range_high']:.2f}/{t['range_low']:.2f}", # Range info
            f"{t['entry_price']:.2f}", f"{t['sl']:.2f}", f"{t['tp']:.2f}", 
            t['res'], f"{t['pnl']:+.2f}", f"{g_equity:,.2f}"
        ])

    print("\n" + "="*120)
    print(f"🗼 TOKYO ORB STRATEGY | Range: 00:00-01:00 UTC | Scan: {args.scan_tf} | RR: {args.rr}")
    print("="*120)
    
    total_trades = len(trades)
    win_rate = (Decimal(stats["LONG"]["win"] + stats["SHORT"]["win"]) / total_trades) * 100
    
    gt = PrettyTable(["Métrique", "Valeur"])
    gt.align = "l"
    gt.add_row(["Capital Final", f"{g_equity:,.2f} $"])
    gt.add_row(["Gain Total (%)", f"{((g_equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100):+.2f} %"])
    gt.add_row(["Trades Totaux", total_trades])
    gt.add_row(["Win Rate Global", f"{win_rate:.2f} %"])
    gt.add_row(["Max Drawdown", f"{g_max_dd_pct:.2f} %"])
    print(gt)

    st = PrettyTable(["Côté", "Trades", "Win Rate", "PnL (R)", "Expectancy"])
    for side in ["LONG", "SHORT"]:
        d = stats[side]
        if d["total"] > 0:
            wr = (Decimal(d["win"]) / d["total"]) * 100
            exp = d["pnl"] / d["total"]
            st.add_row([side, d["total"], f"{wr:.2f}%", f"{d['pnl']:+.2f}R", f"{exp:.3f}R"])
    print(st)

    print("\n📝 JOURNAL (30 DERNIERS TRADES)")
    jt = PrettyTable(["Date", "Side", "Range H/L", "Entry", "SL", "TP", "Res", "PnL", "Equity"])
    for row in journal_rows[-30:]:
        jt.add_row(row)
    print(jt)

    if args.csv:
        filename = f"tokyo_orb_{args.scan_tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        headers = ["Date", "Side", "Range", "Entry", "SL", "TP", "Result", "PnL", "Equity"]
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(journal_rows)
        print(f"\n[+] CSV exporté : {filename}")

# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--scan-tf", default="15m", help="Timeframe de détection (ex: 15m, 1h)")
    ap.add_argument("--rr", type=float, default=4.0)
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()
    
    engine = create_engine(f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}")
    s_ms = int(datetime.strptime(args.start_date, DATE_FORMAT).replace(tzinfo=UTC).timestamp() * 1000)
    
    trades = run_backtest_orb(
        engine, "XAUUSD.c", 
        args.scan_tf,
        Decimal(str(args.rr)), 
        s_ms, 
        int(datetime.now(tz=UTC).timestamp() * 1000)
    )
    display_results(trades, args)

if __name__ == "__main__":
    main()