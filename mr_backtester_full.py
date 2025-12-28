#!/usr/bin/env python3
# postgres_reversion_80PCT_BODY_ULTIMATE_FULL_NO_COMPRESSION.py

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
from typing import Optional, Tuple, Dict, List, Any

# PANDAS POUR LA VITESSE
import pandas as pd
import numpy as np

# Outils de Base de Données
from dotenv import load_dotenv
from sqlalchemy import create_engine

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING (PARAMETRES PAR DEFAUT) ----------
DEFAULT_RR = Decimal("1.0")       
# Pas de limite de temps (No Limit), on attend TP ou SL
SCAN_TF = "30m"                   
EXECUTION_TF_SUFFIX = "1m"        
DEFAULT_RISK_PER_TRADE = Decimal("0.01") 
DEFAULT_FEES_PCT = Decimal("0.1") 

# --- PARAMETRES SPECIFIQUES STRATEGIE REVERSION ---
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULTIPLIER = 3.5      
MIN_BODY_RATIO = 0.8  # Le corps doit faire au moins 80% de la bougie totale
# ------------------------------------------------

# --- PARAMETRES DU SUMMARY ---
INITIAL_BALANCE = Decimal("10000.00") 
GLOBAL_RESULTS = []

# ---------- UTILS BDD & GENERALES ----------

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
    except ValueError:
        raise ValueError(f"Format date invalide {DATE_FORMAT}")

def parse_pairs(path: str):
    out = []
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR")
            if p: out.append(p.strip())
    return out

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}?sslmode=disable",
            pool_pre_ping=True, future=True
        )
        return engine
    except Exception as e:
        print(f"[FATAL] Erreur DB: {e}")
        sys.exit(1)


# --- Fetch Rates (AVEC CALCUL ATR) ---

def fetch_htf_data_pandas(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int], atr_period: int) -> Optional[List[Dict[str, Any]]]:
    """
    Charge les données HTF et calcule l'ATR via Pandas.
    """
    base, quote = pair[:3], pair[3:]
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    query = f"SELECT ts as time, open, high, low, close FROM {table_name}"
    conditions = []
    
    # Buffer pour le calcul de l'ATR
    safe_buffer_ms = timedelta(days=20).total_seconds() * 1000
    
    if start_ms is not None:
        conditions.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms is not None:
        conditions.append(f"ts <= {end_ms}")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY ts ASC"

    try:
        df = pd.read_sql(query, engine)
        if df.empty: return None

        # --- CALCUL ATR (Vectorisé) ---
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = (df['high'] - df['close'].shift(1)).abs()
        df['l-pc'] = (df['low'] - df['close'].shift(1)).abs()
        
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        df['atr'] = df['atr'].fillna(0.0)

        return df[['time', 'open', 'high', 'low', 'close', 'atr']].to_dict('records')

    except Exception as e:
        print(f"[ERR] Pandas Fetch HTF Error for {pair}/{tf}: {e}")
        return None


# --- SIMULATION LTF (SANS LIMITES) ---

def fetch_ltf_data_pandas_full_range(engine, pair: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    # On charge TOUT car pas de limite de temps
    query = f"SELECT ts, high, low FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {end_ms} ORDER BY ts ASC"
    try:
        df = pd.read_sql(query, engine)
        return [] if df.empty else df.to_dict('records')
    except Exception:
        return []

def run_ltf_simulation_standard_no_limit(ltf_data: List[Dict[str, Any]], start_index: int, entry: float, sl: float, tp: float, side: str) -> Tuple[str, int]:
    """
    Exécution STANDARD (Pessimiste: SL vérifié en premier).
    Attend le dénouement (TP ou SL).
    """
    for i in range(start_index, len(ltf_data)):
        row = ltf_data[i]
        ts = row['ts']
        high = float(row['high'])
        low = float(row['low'])
        
        if side == "LONG":
            # PESSIMISTE : On vérifie le SL d'abord
            if low <= sl: return "LOSS", ts
            if high >= tp: return "WIN", ts
            
        elif side == "SHORT":
            # PESSIMISTE : On vérifie le SL d'abord
            if high >= sl: return "LOSS", ts
            if low <= tp: return "WIN", ts
                
    # Fin des données sans résultat
    last_ts = ltf_data[-1]['ts'] if ltf_data else 0
    return "OPEN_AT_END", last_ts


# ----------------------------------------------------------------------
# 🛑 STRATÉGIE : REVERSION (FADE THE MOVE) + FILTRE CORPS 80%
# ----------------------------------------------------------------------

def detect_excess_setup(rates: List[Dict[str, Any]], i: int, scale: int, atr_multiplier: float) -> Optional[Dict[str, Any]]:
    """
    Détecte une bougie anormale avec un CORPS PLEIN (80%) et trade en inverse.
    """
    if i < 1: return None
    
    candle = rates[i]
    atr = candle.get('atr', 0.0)
    if atr == 0: return None

    # 1. Filtre Volatilité (Range Total)
    candle_range = candle['high'] - candle['low']
    if candle_range <= (atr * atr_multiplier):
        return None

    # 2. Filtre Corps (No Doji / No Pinbar)
    open_p = Decimal(str(candle['open']))
    close_p = Decimal(str(candle['close']))
    body_size = abs(close_p - open_p)
    
    # Le corps doit faire 80% de la bougie totale (High - Low)
    if float(candle_range) > 0:
        ratio = float(body_size) / float(candle_range)
        if ratio < MIN_BODY_RATIO:
            return None
    else:
        return None

    # Calcul SL (50% du corps)
    min_sl_dist = Decimal(str(atr)) * Decimal("0.2") 
    calculated_sl_dist = body_size * Decimal("0.5")
    sl_distance = max(calculated_sl_dist, min_sl_dist)

    entry_price = close_p
    
    # REVERSION : Si bougie VERTE -> SHORT | Si bougie ROUGE -> LONG
    if close_p > open_p:
        side = "SHORT"
        sl_price = entry_price + sl_distance 
    else:
        side = "LONG"
        sl_price = entry_price - sl_distance 

    return {
        "side": side,
        "entry_price": qround(entry_price, scale),
        "sl_price": qround(sl_price, scale),
        "atr": atr,
        "candle_range": candle_range
    }

# ----------------------------------------------------------------------


# ---------- LOGIQUE DE BACKTESTING PRINCIPALE ----------

def execute_backtest(engine, pair: str, rr_ratio: Decimal, scale: int, atr_multiplier: float, start_ms: Optional[int], end_ms: Optional[int], risk_per_trade: Decimal, fees_pct: Decimal) -> List[Dict[str, Any]]:
    
    # 1. Data Loading HTF
    rates = fetch_htf_data_pandas(engine, pair, SCAN_TF, start_ms, end_ms, DEFAULT_ATR_PERIOD)
    if not rates or len(rates) < 100: return []
    
    data_start_ms = rates[0]['time']
    data_end_ms = rates[-1]['time']
    
    # 2. Data Loading LTF (Chargement intégral)
    ltf_data = fetch_ltf_data_pandas_full_range(engine, pair, data_start_ms, data_end_ms)
    ltf_timestamps = [r['ts'] for r in ltf_data]
    
    start_index = 0
    if start_ms:
        for idx in range(len(rates)):
            if rates[idx]['time'] >= start_ms:
                start_index = idx
                break
    
    # Init Stats
    balance_r = Decimal(0)
    total_trades = 0; wins = 0; losses = 0
    peak_r = Decimal(0); max_drawdown_r = Decimal(0)
    all_pnl_r = []
    gross_profit_r = Decimal(0); gross_loss_r = Decimal(0)
    trade_log = []
    
    scan_duration_ms = rates[1]['time'] - rates[0]['time'] if len(rates) > 1 else 1800000
    skip_until_ts = 0

    # 3. Boucle Principale
    for i in range(start_index, len(rates)):
        current_ts = rates[i]['time']
        if current_ts < skip_until_ts: continue
        
        # DÉTECTION AVEC FILTRE CORPS 80%
        setup = detect_excess_setup(rates, i, scale, atr_multiplier)
        
        if setup:
            risk_dist = abs(setup["entry_price"] - setup["sl_price"])
            
            # Target calculation
            if setup["side"] == "LONG":
               target_price = setup["entry_price"] + (risk_dist * rr_ratio)
            else: 
               target_price = setup["entry_price"] - (risk_dist * rr_ratio)

            tp_price = qround(target_price, scale)
            
            # Simulation : Démarre à la fin de la bougie signal
            simulation_start_ts = current_ts + scan_duration_ms
            
            ltf_start_idx = bisect.bisect_left(ltf_timestamps, simulation_start_ts)
            
            if ltf_start_idx >= len(ltf_data): continue

            # EXECUTION : STANDARD / NO LIMIT
            result, exit_ts = run_ltf_simulation_standard_no_limit(
                ltf_data, ltf_start_idx, 
                entry=float(setup["entry_price"]), 
                sl=float(setup["sl_price"]), 
                tp=float(tp_price), 
                side=setup["side"]
            )
            
            # Result Processing
            if result in ["WIN", "LOSS"]:
                total_trades += 1
                if result == "WIN":
                    pnl_r = rr_ratio - fees_pct
                    wins += 1
                    gross_profit_r += pnl_r
                else:
                    pnl_r = Decimal("-1.0") - fees_pct
                    losses += 1
                    gross_loss_r += abs(pnl_r)
                
                all_pnl_r.append(float(pnl_r))
                balance_r += pnl_r
                peak_r = max(peak_r, balance_r)
                max_drawdown_r = max(max_drawdown_r, peak_r - balance_r)
                
                trade_log.append({
                    "pair": pair, "entry_time": rates[i]['time'], "exit_time": exit_ts,
                    "side": setup["side"], "entry_price": setup["entry_price"], 
                    "sl_price": setup["sl_price"], "tp_price": tp_price,
                    "result": result, "pnl_r": pnl_r
                })
                skip_until_ts = exit_ts
    
    # Final Stats Calculation
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        expectancy_r = balance_r / total_trades
        profit_factor = gross_profit_r / gross_loss_r if gross_loss_r > 0 else Decimal("99.99")
        
        sqn = 0.0
        if total_trades > 1:
            mean_pnl = statistics.mean(all_pnl_r)
            stdev_pnl = statistics.stdev(all_pnl_r)
            if stdev_pnl > 0: sqn = math.sqrt(total_trades) * (mean_pnl / stdev_pnl)
    else:
        win_rate = 0; expectancy_r = 0; profit_factor = 0; sqn = 0

    GLOBAL_RESULTS.append({
        "pair": pair, "total_trades": total_trades, "wins": wins, "losses": losses,
        "expectancy_r": expectancy_r, "max_drawdown_r": max_drawdown_r,
        "win_rate": win_rate, "profit_factor": profit_factor, "sqn": sqn,
        "start_ts": data_start_ms, "end_ts": data_end_ms
    })
    
    return trade_log

# ---------- AFFICHAGES COMPLETS ----------

def display_summary_table(results: List[Dict[str, Any]]):
    # TRI PAR SQN
    results.sort(key=lambda x: x['sqn'], reverse=True)
    
    # AJOUT TOTAL
    total_trades_all = sum(res['total_trades'] for res in results)
    total_wins_all = sum(res['wins'] for res in results)
    total_losses_all = sum(res['losses'] for res in results)
    global_win_rate = (total_wins_all / total_trades_all) * 100 if total_trades_all > 0 else 0.0
    
    print("\n" + "="*145)
    print(f" SUMMARY BACKTEST - REVERSION (BODY >= 80%) (Scan: {SCAN_TF}, Exe: {EXECUTION_TF_SUFFIX})")
    print("="*145)
    
    header = "| {:^10} | {:^6} | {:^8} | {:^10} | {:^10} | {:^10} | {:^10} | {:^8} | {:^8} |".format(
        "PAIRE", "TRADES", "WIN RATE", "EXPECTANCY", "PROFIT F.", "SQN", "MAX DD(R)", "GAINS", "PERTES"
    )
    separator = "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|"
    
    print(header)
    print(separator)
    
    for res in results:
        print("| {:<10} | {:>6} | {:>9.2f}% | {:>10.4f}R | {:>10.2f} | {:>10.2f} | {:>10.2f}R | {:>8} | {:>8} |".format(
            res['pair'], res['total_trades'], res['win_rate'], float(res['expectancy_r']),
            float(res['profit_factor']), float(res['sqn']), float(res['max_drawdown_r']),
            res['wins'], res['losses']
        ))
    
    print(separator)
    print("| {:<10} | {:>6} | {:>9.2f}% | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} |".format(
        "TOTAL", total_trades_all, global_win_rate, "", "", "", "", total_wins_all, total_losses_all
    ))
    print("="*145 + "\n")

def display_hourly_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    hourly_stats = {h: {'wins': 0, 'losses': 0, 'total': 0} for h in range(24)}
    has_trades = False
    
    for pair, trades in all_trades_log.items():
        for trade in trades:
            has_trades = True
            entry_ts = trade['entry_time']
            result = trade['result']
            dt = datetime.fromtimestamp(entry_ts / 1000, tz=UTC)
            hour = dt.hour
            hourly_stats[hour]['total'] += 1
            if result == "WIN": hourly_stats[hour]['wins'] += 1
            elif result == "LOSS": hourly_stats[hour]['losses'] += 1
    
    if not has_trades: return
    print("\n" + "="*50)
    print(" 🕒 HOURLY BREAKDOWN (UTC TIME)")
    print("="*50)
    print("| {:^6} | {:^8} | {:^8} | {:^10} |".format("HOUR", "TRADES", "WINS", "WIN RATE"))
    print("|" + "-"*8 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|")
    
    for h in range(24):
        stats = hourly_stats[h]
        total = stats['total']
        wr = (stats['wins'] / total * 100) if total > 0 else 0.0
        print("| {:02d}:00  | {:^8} | {:^8} | {:>9.2f}% |".format(h, total, stats['wins'], wr))
    print("="*50 + "\n")

def display_daily_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    daily_stats = {d: {'wins': 0, 'losses': 0, 'total': 0} for d in range(7)}
    day_names = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]
    has_trades = False
    
    for pair, trades in all_trades_log.items():
        for trade in trades:
            has_trades = True
            dt = datetime.fromtimestamp(trade['entry_time'] / 1000, tz=UTC)
            daily_stats[dt.weekday()]['total'] += 1
            if trade['result'] == "WIN": daily_stats[dt.weekday()]['wins'] += 1
            elif trade['result'] == "LOSS": daily_stats[dt.weekday()]['losses'] += 1
    
    if not has_trades: return
    print("\n" + "="*50)
    print(" 📅 DAILY BREAKDOWN (UTC TIME)")
    print("="*50)
    print("| {:^10} | {:^8} | {:^8} | {:^10} |".format("DAY", "TRADES", "WINS", "WIN RATE"))
    print("|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|")
    
    for d in range(7):
        stats = daily_stats[d]
        total = stats['total']
        wr = (stats['wins'] / total * 100) if total > 0 else 0.0
        print("| {:<10} | {:^8} | {:^8} | {:>9.2f}% |".format(day_names[d], total, stats['wins'], wr))
    print("="*50 + "\n")

def get_asset_type(pair: str) -> str:
    p_up = pair.upper()
    if any(k in p_up for k in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']): return "CRYPTO"
    if any(k in p_up for k in ['US30', 'SPX', 'NAS100', 'DAX', 'GER30']): return "INDICES"
    if any(k in p_up for k in ['XAU', 'XAG', 'WTI', 'OIL']): return "COMMODITIES"
    return "FOREX"

def display_keepers_csv(results: List[Dict[str, Any]]):
    print("\n" + "="*80)
    print(" 💎 KEEPER PAIRS (FILTRE: PF >= 1.2 & Trades >= 20)")
    print("="*80)
    keepers = [r for r in results if r['total_trades'] >= 20 and r['profit_factor'] >= 1.2]
    keepers.sort(key=lambda x: x['sqn'], reverse=True)
    print("type,pair")
    for res in keepers:
        print(f"{get_asset_type(res['pair'])},{res['pair']}")
    print("\n")

def display_portfolio_simulation(all_trades_log: Dict[str, List[Dict[str, Any]]], initial_capital: Decimal, risk_per_trade: Decimal):
    all_trades = []
    for pair, trades in all_trades_log.items():
        for t in trades:
            trade_copy = t.copy()
            trade_copy['pair'] = pair
            all_trades.append(trade_copy)
            
    if not all_trades: return
    all_trades.sort(key=lambda x: x['exit_time'])

    current_balance = initial_capital
    high_water_mark = initial_capital
    max_dd_amt = Decimal(0); max_dd_pct = Decimal(0)
    
    print("\n" + "="*60)
    print(" 💰 PORTFOLIO SIMULATION")
    print("="*60)
    
    for t in all_trades:
        risk_amt = current_balance * risk_per_trade
        pnl = risk_amt * t['pnl_r']
        current_balance += pnl
        
        if current_balance > high_water_mark: high_water_mark = current_balance
        dd = high_water_mark - current_balance
        if dd > max_dd_amt: max_dd_amt = dd
        if high_water_mark > 0:
            dd_pct = (dd / high_water_mark) * 100
            if dd_pct > max_dd_pct: max_dd_pct = dd_pct
            
    roi = ((current_balance - initial_capital) / initial_capital) * 100
    print(f"{'CAPITAL INITIAL':<30} : {initial_capital:,.2f}")
    print(f"{'CAPITAL FINAL':<30} : {current_balance:,.2f}")
    print(f"{'NET PROFIT':<30} : {roi:+.2f}%")
    print("-" * 60)
    print(f"{'MAX DRAWDOWN (%)':<30} : -{max_dd_pct:.2f}%")
    print(f"{'TOTAL TRADES':<30} : {len(all_trades)}")
    print("="*60 + "\n")

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--rr", type=Decimal, default=DEFAULT_RR, help="Ratio Risk/Reward")
    ap.add_argument("--atr-multiplier", type=float, default=DEFAULT_ATR_MULTIPLIER, help="Seuil d'anomalie (x ATR)")
    ap.add_argument("--risk", type=Decimal, default=DEFAULT_RISK_PER_TRADE)
    ap.add_argument("--fees", type=Decimal, default=DEFAULT_FEES_PCT)
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--end-date", type=str, default=None)
    
    args = ap.parse_args()
    
    start_ms = parse_date_to_ms(args.start_date) if args.start_date else None
    end_ms = parse_date_to_ms(args.end_date, is_end_date=True) if args.end_date else None
    
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    
    print(f"Strategy REVERSION (No Limit) | Body Ratio >= {MIN_BODY_RATIO*100}% | RR: {args.rr}")
    
    all_trades_log = {} 

    for p in pairs:
        base, quote = p[:3], p[3:]
        scale = price_scale(base, quote)
        trade_log = execute_backtest(engine, p, args.rr, scale, args.atr_multiplier, start_ms, end_ms, args.risk, args.fees)
        all_trades_log[p] = trade_log

    # 1. Résumé par paire
    display_summary_table(GLOBAL_RESULTS)
    
    # 2. Stats horaires
    display_hourly_breakdown(all_trades_log)
    
    # 3. Stats journalières
    display_daily_breakdown(all_trades_log)
    
    # 4. Keepers (CSV)
    display_keepers_csv(GLOBAL_RESULTS)
    
    # 5. Simulation Portefeuille
    display_portfolio_simulation(all_trades_log, INITIAL_BALANCE, args.risk)

if __name__ == "__main__":
    main()