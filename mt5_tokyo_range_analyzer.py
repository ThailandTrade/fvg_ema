#!/usr/bin/env python3
# pg_tokyo_dual_source.py
"""
Analyse PostgreSQL : Session Tokyo (15m) vs Range J-1 (1d).
Source 1 : Table 'candles_mt5_{pair}_15m' pour la session de Tokyo.
Source 2 : Table 'candles_mt5_{pair}_1d' pour les High/Low du jour précédent.
"""

import os, re, csv, sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# --- CONFIGURATION ---
TOKYO_START_HOUR = 0
TOKYO_END_HOUR = 9   # 09:00 UTC exclu (donc 00:00 à 08:59:59)

def get_pg_engine():
    load_dotenv()
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "postgres")
    return create_engine(
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=disable",
        future=True
    )

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def parse_pairs(path: str):
    out = []
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p:
                out.append(p.strip())
    return out

def check_tables_exist(engine, pair):
    insp = inspect(engine)
    t15 = f"candles_mt5_{sanitize_name(pair)}_15m"
    t1d = f"candles_mt5_{sanitize_name(pair)}_1d"
    existing = insp.get_table_names()
    return (t15 in existing, t1d in existing, t15, t1d)

def analyze_pair_dual(engine, pair):
    exists_15, exists_1d, table_15m, table_1d = check_tables_exist(engine, pair)
    
    if not exists_15 or not exists_1d:
        print(f"[SKIP] {pair}: Tables manquantes (15m: {exists_15}, 1d: {exists_1d})")
        return None, None

    # --- 1. CHARGEMENT DONNÉES DAILY (Table 1d) ---
    try:
        q_d = f"SELECT ts, high, low FROM \"{table_1d}\" ORDER BY ts ASC"
        df_daily = pd.read_sql(q_d, engine)
    except Exception as e:
        print(f"[ERR] Lecture {table_1d}: {e}")
        return None, None

    if df_daily.empty:
        return None, None

    # Conversion TS -> Date Index
    df_daily['datetime'] = pd.to_datetime(df_daily['ts'], unit='ms', utc=True)
    # On arrondit à la date (minuit) pour assurer la jointure
    df_daily['date'] = df_daily['datetime'].dt.date 
    df_daily.set_index('date', inplace=True)
    
    # CALCUL DU RANGE J-1
    # On décale les colonnes High/Low vers le bas de 1.
    # Ainsi, à la date du "2023-10-02", on aura les valeurs du "2023-10-01".
    df_prev = df_daily[['high', 'low']].shift(1).rename(columns={'high': 'prev_high', 'low': 'prev_low'})
    df_prev['prev_range'] = df_prev['prev_high'] - df_prev['prev_low']
    
    # Nettoyage des jours sans historique précédent (le premier jour)
    df_prev.dropna(inplace=True)

    # --- 2. CHARGEMENT DONNÉES TOKYO (Table 15m) ---
    try:
        # On ne charge que ce qu'il faut pour économiser la mémoire si la base est grosse
        q_m = f"SELECT ts, high, low FROM \"{table_15m}\" ORDER BY ts ASC"
        df_15 = pd.read_sql(q_m, engine)
    except Exception as e:
        print(f"[ERR] Lecture {table_15m}: {e}")
        return None, None

    if df_15.empty:
        return None, None

    df_15['datetime'] = pd.to_datetime(df_15['ts'], unit='ms', utc=True)
    
    # FILTRAGE SESSION TOKYO (00:00 <= Heure < 09:00 UTC)
    mask_tokyo = (df_15['datetime'].dt.hour >= TOKYO_START_HOUR) & (df_15['datetime'].dt.hour < TOKYO_END_HOUR)
    df_tokyo_raw = df_15[mask_tokyo].copy()

    if df_tokyo_raw.empty:
        print(f"[SKIP] {pair}: Pas de données sur la plage horaire Tokyo.")
        return None, None

    # Agrégation par jour pour avoir le High/Low de la session Tokyo de ce jour là
    df_tokyo_raw['date'] = df_tokyo_raw['datetime'].dt.date
    df_tokyo_stats = df_tokyo_raw.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).rename(columns={'high': 'tok_high', 'low': 'tok_low'})
    
    df_tokyo_stats['tok_range'] = df_tokyo_stats['tok_high'] - df_tokyo_stats['tok_low']

    # --- 3. FUSION & ANALYSE ---
    # Inner join sur la date : on ne garde que les jours où on a les stats Tokyo ET l'historique J-1
    merged = df_tokyo_stats.join(df_prev, how='inner')

    if merged.empty:
        return None, None

    # Logique INSIDE : Tokyo entièrement inclus dans le range J-1
    merged['is_inside'] = (merged['tok_high'] < merged['prev_high']) & (merged['tok_low'] > merged['prev_low'])
    
    # Logique BREAKOUT
    merged['break_high'] = merged['tok_high'] >= merged['prev_high']
    merged['break_low']  = merged['tok_low'] <= merged['prev_low']
    
    # Logique COMPRESSION
    merged['compression_ratio'] = (merged['tok_range'] / merged['prev_range']) * 100

    # Stats finales
    total_days = len(merged)
    inside_count = merged['is_inside'].sum()
    break_h = merged['break_high'].sum()
    break_l = merged['break_low'].sum()
    avg_comp = merged['compression_ratio'].mean()

    inside_pct = (inside_count / total_days) * 100
    
    print(f"{pair:<10} | Days: {total_days:<4} | Inside: {inside_pct:5.1f}% | Ratio: {avg_comp:5.1f}% | Break H: {(break_h/total_days)*100:4.0f}% | Break L: {(break_l/total_days)*100:4.0f}%")

    return inside_pct, avg_comp

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    args = ap.parse_args()

    engine = get_pg_engine()

    pairs = parse_pairs(args.pairs_file)
    if not pairs:
        print("Fichier pairs.txt vide ou introuvable.")
        sys.exit(1)

    print(f"--- ANALYSE DUAL TABLE (1d vs 15m) ---")
    print(f"Ref Daily: Table *_1d (High/Low J-1)")
    print(f"Ref Tokyo: Table *_15m ({TOKYO_START_HOUR}h-{TOKYO_END_HOUR}h UTC du Jour J)")
    print("-" * 85)
    print(f"{'PAIR':<10} | {'DAYS':<5} | {'INSIDE %':<13} | {'RATIO %':<12} | {'DETAILS'}")
    print("-" * 85)

    global_inside = []
    global_ratio = []

    for p in pairs:
        i_pct, r_pct = analyze_pair_dual(engine, p)
        if i_pct is not None:
            global_inside.append(i_pct)
            global_ratio.append(r_pct)

    if global_inside:
        print("-" * 85)
        avg_ins = sum(global_inside) / len(global_inside)
        avg_rat = sum(global_ratio) / len(global_ratio)
        print(f"{'MOYENNE':<10} | {'ALL':<5} | Inside: {avg_ins:5.1f}% | Ratio: {avg_rat:5.1f}%")

if __name__ == "__main__":
    main()