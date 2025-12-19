#!/usr/bin/env python3
# pg_backtest_tokyo_ema_filter.py
"""
BACKTEST TOKYO EMA FILTER.
- Filtre : EMA 50 (par défaut).
- Logique : Rebond PDH/PDL UNIQUEMENT si aligné avec la tendance EMA.
- Entrées : 00h00 - 09h00 UTC.
- Correction : Utilise le High/Low de J-2 (lag=2).
"""

import os, re, csv, sys
import argparse
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

# --- CONFIGURATION ---
TOKYO_END_HOUR = 9  # Fin des entrées à 09h00 UTC

def get_pg_engine():
    load_dotenv()
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "postgres")
    return create_engine(
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db}?sslmode=disable",
        future=True,
        connect_args={'client_encoding': 'utf8'}
    )

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def parse_pairs(path: str):
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r.get("pair") or r.get("PAIR") for r in csv.DictReader(f) if r.get("pair") or r.get("PAIR")]

def run_ema_backtest(engine, pair, start_date, rr, sl_pct, lag_days, ema_period):
    t_15m = f"candles_mt5_{sanitize_name(pair)}_15m"
    t_1d  = f"candles_mt5_{sanitize_name(pair)}_1d"
    
    # 1. CHARGEMENT DONNÉES
    try:
        df_d = pd.read_sql(f"SELECT ts, high, low FROM \"{t_1d}\" ORDER BY ts ASC", engine)
        df_m = pd.read_sql(f"SELECT ts, open, high, low, close FROM \"{t_15m}\" ORDER BY ts ASC", engine)
    except:
        return None

    if df_d.empty or df_m.empty: return None

    # Conversion Dates
    df_d['date'] = pd.to_datetime(df_d['ts'], unit='ms', utc=True).dt.date
    df_m['datetime'] = pd.to_datetime(df_m['ts'], unit='ms', utc=True)
    df_m['date'] = df_m['datetime'].dt.date
    df_m['hour'] = df_m['datetime'].dt.hour
    
    if start_date:
        s_dt = pd.to_datetime(start_date).date()
        df_d = df_d[df_d['date'] >= s_dt]
        df_m = df_m[df_m['date'] >= s_dt]

    # --- 2. CALCUL EMA ---
    df_m['ema'] = df_m['close'].ewm(span=ema_period, adjust=False).mean()


    # 3. APPLICATION DU LAG (CORRECTION DB)
    df_d = df_d.set_index('date').sort_index()
    levels = df_d[['high', 'low']].shift(lag_days).dropna()
    levels.rename(columns={'high': 'pdh', 'low': 'pdl'}, inplace=True)
    levels['prev_range'] = levels['pdh'] - levels['pdl']

    # 4. MERGE
    df = df_m.join(levels, on='date', how='inner')
    if df.empty: return None

    # 5. EXÉCUTION (State Machine)
    trades = []
    active_trade = None
    last_trade_day = None 
    
    # On commence après la période de chauffe de l'EMA
    start_idx = ema_period 
    subset = df.iloc[start_idx:]
    
    for row in subset.itertuples():
        
        # --- GESTION DES TRADES OUVERTS ---
        if active_trade:
            res = None
            pnl = 0.0
            
            # Vérification SL/TP
            if active_trade['type'] == 'SHORT':
                if row.high >= active_trade['sl']:
                    res = 'loss'
                    pnl = -1.0
                elif row.low <= active_trade['tp']:
                    res = 'win'
                    pnl = rr
            else: # LONG
                if row.low <= active_trade['sl']:
                    res = 'loss'
                    pnl = -1.0
                elif row.high >= active_trade['tp']:
                    res = 'win'
                    pnl = rr
            
            # Force Close en fin de journée (23h)
            if res is None and row.hour >= 23:
                res = 'flat'
                if active_trade['type'] == 'SHORT':
                    pnl = (active_trade['entry'] - row.close) / (active_trade['sl'] - active_trade['entry'])
                else:
                    pnl = (row.close - active_trade['entry']) / (active_trade['entry'] - active_trade['sl'])

            if res:
                trades.append({'res': res, 'pnl': pnl})
                active_trade = None
            
            continue

        # --- RECHERCHE D'ENTRÉE (TOKYO SEULEMENT) ---
        
        if last_trade_day == row.date: continue
        if row.hour >= TOKYO_END_HOUR: continue
        
        pdh = row.pdh
        pdl = row.pdl
        rng = row.prev_range
        ema = row.ema # Valeur de l'EMA 50
        
        if pdh != pdh or rng == 0: continue
        if ema != ema: continue # Check NaN sur EMA
        
        sl_dist = rng * sl_pct
        
        # FILTRES DE TENDANCE
        trend_is_up = row.close > ema
        trend_is_down = row.close < ema

        # 1. SETUP SHORT (Prix touche PDH ET Tendance Baissière)
        if row.high >= pdh and trend_is_down:
            entry = pdh
            sl = pdh + sl_dist
            
            if row.high >= sl:
                trades.append({'res': 'loss', 'pnl': -1.0})
                last_trade_day = row.date
            else:
                active_trade = {
                    'type': 'SHORT', 'entry': entry, 'sl': sl, 
                    'tp': entry - (sl-entry)*rr
                }
                last_trade_day = row.date

        # 2. SETUP LONG (Prix touche PDL ET Tendance Haussière)
        elif row.low <= pdl and trend_is_up:
            entry = pdl
            sl = pdl - sl_dist
            
            if row.low <= sl:
                trades.append({'res': 'loss', 'pnl': -1.0})
                last_trade_day = row.date
            else:
                active_trade = {
                    'type': 'LONG', 'entry': entry, 'sl': sl, 
                    'tp': entry + (entry-sl)*rr
                }
                last_trade_day = row.date

    return trades

def print_stats(pair, trades):
    if not trades: return 0.0
    df_t = pd.DataFrame(trades)
    
    total = len(df_t)
    wins = len(df_t[df_t['pnl'] > 0])
    wr = (wins/total)*100
    pnl = df_t['pnl'].sum()
    
    # Drawdown
    df_t['cum'] = df_t['pnl'].cumsum()
    dd = df_t['cum'] - df_t['cum'].cummax()
    max_dd = dd.min()

    c = "\033[92m" if pnl > 0 else "\033[91m"
    r = "\033[0m"
    
    print(f"{pair:<10} | Trades: {total:<4} | WR: {wr:5.1f}% | PnL: {c}{pnl:6.1f}R{r} | MaxDD: {max_dd:5.1f}R")
    return pnl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--sl-pct", type=float, default=0.10)
    ap.add_argument("--lag", type=int, default=2, help="Correction DB: J-2")
    ap.add_argument("--ema", type=int, default=50, help="Période EMA (Filtre Tendance)")
    args = ap.parse_args()

    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    
    print(f"--- BACKTEST TOKYO + EMA {args.ema} FILTER ---")
    print(f"Session: 00h-09h UTC | Lag DB: J-{args.lag} | RR: {args.rr}")
    print("-" * 75)
    print(f"{'PAIR':<10} | {'TRADES':<6} | {'WIN RATE':<9} | {'PnL (R)':<10} | {'MAX DD'}")
    print("-" * 75)

    port_pnl = 0.0
    for p in pairs:
        trades = run_ema_backtest(engine, p, args.start, args.rr, args.sl_pct, args.lag, args.ema)
        if trades:
            port_pnl += print_stats(p, trades)
    
    print("-" * 75)
    print(f"PORTFOLIO TOTAL: {port_pnl:.1f} R")

if __name__ == "__main__":
    main()