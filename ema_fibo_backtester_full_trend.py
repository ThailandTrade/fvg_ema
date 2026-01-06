#!/usr/bin/env python3
# postgres_fvg_backtester_PANDAS_FULL_FINAL_V4_HTF.py

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

# NOUVEAU : PANDAS POUR LA VITESSE
import pandas as pd
import numpy as np

# Outils de Base de Données
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, select, desc, inspect, and_, text
from sqlalchemy.types import Numeric

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING (PARAMETRES ICI) ----------
DEFAULT_RR = Decimal("2.0")
MAX_WAIT_CANDLES = 32
SCAN_TF = "5m"            # Unité de temps pour la détection du setup
TREND_FILTER_TF = "1h"   # NOUVEAU : Timeframe du filtre de tendance (Option C)
EXECUTION_TF_SUFFIX = "1m" # Timeframe pour l'exécution précise
DEFAULT_RISK_PER_TRADE = Decimal("0.003")
DEFAULT_FEES_PCT = Decimal("0.01") # 5% de frais par trade (calculé sur le risque 1R)

# --- FILTRES DE DIRECTION (Par défaut) ---
DEFAULT_ALLOW_LONG = True
DEFAULT_ALLOW_SHORT = False

# --- PARAMETRES DE LA NOUVELLE STRATEGIE ---
EMA_TREND_PERIOD = 200
FIB_RETREACEMENT = 0.62
SWING_CONFIRMATION_LAG = 5 

# --- PARAMETRES DU SUMMARY ---
INITIAL_BALANCE = Decimal("100000.00") # Capital de départ pour la simulation
SHOW_ALL_TRADES = True                # Mettre à True pour voir la liste complète, False pour les 3 derniers

# ---------- CONSTANTES POUR STDEV ----------
STDEV_PERIOD = 200 
DEFAULT_STDEV_THRESHOLD = 0.5
DEFAULT_STDEV_MAX = 1.0
# ----------------------------------------

# --- Structure pour les résultats globaux ---
GLOBAL_RESULTS = []

# ---------- UTILS BDD & GENERALES (Inchangées) ----------

def price_scale(base: str, quote: str) -> int:
    # Détermine l'échelle (nombre de décimales) pour le prix
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    # Arrondi quantifié
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    # Formatage du timestamp en date/heure UTC
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime('%m-%d %H:%M')

def parse_date_to_ms(date_str: str, is_end_date: bool = False) -> int:
    """Convertit une date string (AAAA-MM-JJ) en timestamp UTC en millisecondes."""
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT).replace(tzinfo=UTC)
        if is_end_date:
            dt += timedelta(days=1) - timedelta(milliseconds=1)
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(f"Le format de date doit être {DATE_FORMAT} (Ex: 2024-01-15)")

def parse_pairs(path: str):
    # Lecture des paires depuis un fichier CSV/TXT
    out = []
    if not os.path.exists(path):
        print(f"[WARN] Fichier de paires non trouvé: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p:
                out.append(p.strip())
    return out

def sanitize_name(s: str) -> str:
    # Nettoyage pour les noms de table SQL
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    # Connexion à la base de données PostgreSQL
    load_dotenv()
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "postgres")
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=disable",
            pool_pre_ping=True,
            future=True
        )
        with engine.connect():
            pass 
        return engine
    except Exception as e:
        print(f"[FATAL] Échec de la connexion PostgreSQL. Vérifiez .env et le service : {e}")
        sys.exit(1)


# --- NOUVEAU : FETCH HTF TREND STRUCTURE (30m) ---

def fetch_trend_structure_data(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> pd.DataFrame:
    """
    Charge les données du filtre HTF (ex: 30m), calcule les pivots et définit la tendance structurelle.
    Retourne un DataFrame avec colonnes ['time', 'htf_trend'].
    htf_trend: 1 (Uptrend), -1 (Downtrend), 0 (Neutre)
    """
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"
    query = f"SELECT ts as time, high, low, close FROM {table_name}"
    conditions = []
    
    # Buffer plus large car le HTF est plus lent
    safe_buffer_ms = timedelta(days=60).total_seconds() * 1000
    
    if start_ms is not None:
        conditions.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms is not None:
        conditions.append(f"ts <= {end_ms}")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY ts ASC"

    try:
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()

        # --- Détection des Swings ---
        l = SWING_CONFIRMATION_LAG
        window_size = 2 * l + 1
        
        rolling_max = df['high'].rolling(window=window_size).max()
        rolling_min = df['low'].rolling(window=window_size).min()
        
        # On identifie les swings bruts (à leur place réelle)
        # Shift(-l) car rolling regarde en arrière, mais le sommet est au milieu
        # Note: Pandas rolling aligne à droite. Pour centrer: center=True ou shift. 
        # Ici on garde la logique précédente : max sur window, check si milieu == max
        
        df['is_swing_high'] = False
        df['is_swing_low'] = False
        
        # Optimisation vectorisée approximative pour vitesse ou boucle pour précision
        # On va utiliser une boucle rapide sur les index détectés
        
        # Pour faire simple et robuste comme avant:
        for idx in range(window_size, len(df)):
            mid_idx = idx - l
            if df.iloc[mid_idx]['high'] == rolling_max.iloc[idx]:
                df.at[mid_idx, 'is_swing_high'] = True
            if df.iloc[mid_idx]['low'] == rolling_min.iloc[idx]:
                df.at[mid_idx, 'is_swing_low'] = True

        # --- Calcul de la Structure (Option C) avec respect du LAG ---
        # On itère pour définir la tendance. Attention: un swing n'est "connu" qu'à (index + l)
        
        last_h = -1.0; prev_h = -1.0
        last_l = -1.0; prev_l = -1.0
        current_trend = 0 # 0: Neutre, 1: Bull, -1: Bear
        
        trend_col = np.zeros(len(df), dtype=int)
        
        # On convertit en numpy pour itération ultra rapide
        highs = df['high'].values
        lows = df['low'].values
        is_sh = df['is_swing_high'].values
        is_sl = df['is_swing_low'].values
        
        for i in range(len(df)):
            # 1. Vérifier si un swing a été CONFIRMÉ à cette bougie 'i'
            # Le swing s'est produit à i - l
            confirmed_idx = i - l
            
            if confirmed_idx >= 0:
                if is_sh[confirmed_idx]:
                    prev_h = last_h
                    last_h = highs[confirmed_idx]
                    
                if is_sl[confirmed_idx]:
                    prev_l = last_l
                    last_l = lows[confirmed_idx]
                
                # 2. Mise à jour de la tendance basée sur les structures connues
                if last_h > 0 and prev_h > 0 and last_l > 0 and prev_l > 0:
                    # Higher Highs AND Higher Lows -> Uptrend
                    if last_h > prev_h and last_l > prev_l:
                        current_trend = 1
                    # Lower Lows AND Lower Highs -> Downtrend
                    elif last_l < prev_l and last_h < prev_h:
                        current_trend = -1
                    # Sinon on garde la tendance précédente ou 0 si cassure mixte (range)
            
            trend_col[i] = current_trend

        df['htf_trend'] = trend_col
        df['htf_trend'] = df['htf_trend'].shift(1).fillna(0)
        return df[['time', 'htf_trend']]

    except Exception as e:
        print(f"[ERR] Trend Filter Data Error for {pair}: {e}")
        return pd.DataFrame()


# --- Fetch Rates (ADAPTÉ POUR RETOURNER DF POUR MERGE) ---

def fetch_htf_data_pandas_raw(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> pd.DataFrame:
    """
    Même logique que l'original mais retourne le DataFrame Pandas brut au lieu d'une liste.
    """
    base, quote = pair[:3], pair[3:]
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    query = f"SELECT ts as time, open, high, low, close FROM {table_name}"
    conditions = []
    
    safe_buffer_ms = timedelta(days=40).total_seconds() * 1000
    
    if start_ms is not None:
        conditions.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms is not None:
        conditions.append(f"ts <= {end_ms}")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY ts ASC"

    try:
        df = pd.read_sql(query, engine)
        if df.empty: return pd.DataFrame()

        # EMA 200
        df['ema_trend'] = df['close'].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()

        # SWINGS
        l = SWING_CONFIRMATION_LAG
        window_size = 2 * l + 1
        rolling_max = df['high'].rolling(window=window_size).max()
        rolling_min = df['low'].rolling(window=window_size).min()
        
        df['is_swing_high'] = False
        df['is_swing_low'] = False
        
        for idx in range(window_size, len(df)):
            mid_idx = idx - l
            if df.iloc[mid_idx]['high'] == rolling_max.iloc[idx]:
                df.at[mid_idx, 'is_swing_high'] = True
            if df.iloc[mid_idx]['low'] == rolling_min.iloc[idx]:
                df.at[mid_idx, 'is_swing_low'] = True

        return df

    except Exception as e:
        print(f"[ERR] Pandas Fetch HTF Error for {pair}/{tf}: {e}")
        return pd.DataFrame()


# --- SIMULATION LTF (Inchangée) ---

def fetch_ltf_data_pandas(engine, pair: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    buffer_end = end_ms + (MAX_WAIT_CANDLES * 30 * 60 * 1000 * 2) 
    query = f"SELECT ts, high, low FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {buffer_end} ORDER BY ts ASC"
    try:
        df = pd.read_sql(query, engine)
        if df.empty: return []
        return df.to_dict('records')
    except Exception as e:
        return []

def run_ltf_simulation_memory(ltf_data: List[Dict[str, Any]], start_index: int, entry: float, sl: float, tp: float, side: str, expiration_ts: int) -> Tuple[str, int, int]:
    is_open = False
    real_entry_ts = 0 
    max_steps = 10000 
    end_loop = min(start_index + max_steps, len(ltf_data))
    
    for i in range(start_index, end_loop):
        row = ltf_data[i]
        ts = row['ts']
        high = float(row['high'])
        low = float(row['low'])
        
        if not is_open:
            if ts > expiration_ts: return "EXPIRED", 0, ts
            
            if side == "LONG":
                if low <= entry:
                    is_open = True
                    real_entry_ts = ts
                    if low <= sl: return "LOSS", real_entry_ts, ts
            elif side == "SHORT":
                if high >= entry:
                    is_open = True
                    real_entry_ts = ts
                    if high >= sl: return "LOSS", real_entry_ts, ts
        
        if is_open:
            if side == "LONG":
                if low <= sl: return "LOSS", real_entry_ts, ts
                if high >= tp: return "WIN", real_entry_ts, ts
            elif side == "SHORT":
                if high >= sl: return "LOSS", real_entry_ts, ts
                if low <= tp: return "WIN", real_entry_ts, ts
                
    if ltf_data:
        last_ts = ltf_data[end_loop - 1]['ts']
    else:
        last_ts = expiration_ts
        
    return "EXPIRED", 0, last_ts


# --- DÉTECTION DE SETUP (MODIFIÉ POUR FILTRE HTF) ---

def detect_fvg_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_threshold: float, stdev_max: float, allow_longs: bool, allow_shorts: bool) -> Optional[Dict[str, Any]]:
    """
    Détection EMA 200 + Fibo + NOUVEAU Filtre HTF Trend.
    """
    vision_limit = i - SWING_CONFIRMATION_LAG
    if vision_limit < 200: return None

    curr = rates[i]
    ema = curr.get('ema_trend')
    htf_trend = curr.get('htf_trend', 0) # Récupération de la tendance HTF fusionnée

    if ema is None or pd.isna(ema): return None

    is_uptrend_local = curr['close'] > ema
    is_downtrend_local = curr['close'] < ema

    # --- RECHERCHE LONG ---
    # Condition ajoutée : and htf_trend == 1
    if allow_longs and is_uptrend_local and htf_trend == 1:
        sh_idx = -1
        for k in range(vision_limit, vision_limit - 60, -1):
            if rates[k]['is_swing_high']:
                sh_idx = k; break
        if sh_idx == -1: return None

        sl_idx = -1
        for k in range(sh_idx - 1, sh_idx - 100, -1):
            if rates[k]['is_swing_low']:
                sl_idx = k; break
        if sl_idx == -1: return None

        high_p = rates[sh_idx]['high']
        low_p = rates[sl_idx]['low']
        fib_price = high_p - ((high_p - low_p) * FIB_RETREACEMENT)

        if fib_price > ema and curr['close'] > fib_price:
            return {
                "side": "LONG",
                "entry_price": qround(fib_price, scale),
                "sl_price": qround(low_p, scale),
                "stdev_score": 0.0, "gap_size": 0.0
            }

    # --- RECHERCHE SHORT ---
    # Condition ajoutée : and htf_trend == -1
    elif allow_shorts and is_downtrend_local and htf_trend == -1:
        sl_idx = -1
        for k in range(vision_limit, vision_limit - 60, -1):
            if rates[k]['is_swing_low']:
                sl_idx = k; break
        if sl_idx == -1: return None

        sh_idx = -1
        for k in range(sl_idx - 1, sl_idx - 100, -1):
            if rates[k]['is_swing_high']:
                sh_idx = k; break
        if sh_idx == -1: return None

        low_p = rates[sl_idx]['low']
        high_p = rates[sh_idx]['high']
        fib_price = low_p + ((high_p - low_p) * FIB_RETREACEMENT)

        if fib_price < ema and curr['close'] < fib_price:
            return {
                "side": "SHORT",
                "entry_price": qround(fib_price, scale),
                "sl_price": qround(high_p, scale),
                "stdev_score": 0.0, "gap_size": 0.0
            }

    return None


# ---------- LOGIQUE DE BACKTESTING PRINCIPALE (MODIFIÉE POUR MERGE) ----------

def execute_backtest(engine, pair: str, rr_ratio: Decimal, scale: int, stdev_threshold: float, start_ms: Optional[int], end_ms: Optional[int], risk_per_trade: Decimal, stdev_max: float, fees_pct: Decimal, allow_longs: bool, allow_shorts: bool) -> List[Dict[str, Any]]:
    
    # 1. CHARGEMENT DONNÉES SCAN (5m) EN DATAFRAME
    df_scan = fetch_htf_data_pandas_raw(engine, pair, SCAN_TF, start_ms, end_ms)
    if df_scan.empty or len(df_scan) < 200: return []
    
    # 2. CHARGEMENT DONNÉES FILTRE HTF (30m)
    df_filter = fetch_trend_structure_data(engine, pair, TREND_FILTER_TF, start_ms, end_ms)
    
    # 3. FUSION (MERGE) INTELLIGENTE
    # On utilise merge_asof pour associer la tendance 30m la plus récente à chaque bougie 5m
    # direction='backward' assure qu'on prend la valeur connue dans le passé ou présent immédiat
    if not df_filter.empty:
        df_scan = pd.merge_asof(
            df_scan.sort_values('time'),
            df_filter.sort_values('time'),
            on='time',
            direction='backward'
        )
        # Remplir les NaN (au début avant le premier calcul HTF) par 0 (Neutre)
        df_scan['htf_trend'] = df_scan['htf_trend'].fillna(0)
    else:
        # Si pas de data filter (erreur?), on met tout à 0 (pas de trade) ou on ignore le filtre ?
        # Ici on met 0, donc aucun trade ne sera pris si le filtre échoue
        df_scan['htf_trend'] = 0

    # Conversion en liste de dicts pour l'itération existante
    rates = df_scan.to_dict('records')

    data_start_ms = rates[0]['time']
    data_end_ms = rates[-1]['time']
    
    # 4. CHARGEMENT LTF (1m)
    ltf_data = fetch_ltf_data_pandas(engine, pair, data_start_ms, data_end_ms)
    ltf_timestamps = [r['ts'] for r in ltf_data]
    
    # 5. FILTRAGE DE LA PLAGE DE SCAN
    start_index = 0
    end_index = len(rates)
    required_seed = 200 
    
    if start_ms is not None:
        for idx in range(required_seed, len(rates)):
            if rates[idx]['time'] >= start_ms:
                start_index = idx
                break
        else: return []

    if end_ms is not None:
        for idx in range(start_index, len(rates)):
            if rates[idx]['time'] > end_ms:
                end_index = idx
                break

    if end_index <= start_index: return []
    
    summary_start_ts = rates[start_index]['time']
    summary_end_ts = rates[end_index - 1]['time']
    
    balance_r = Decimal(0)
    total_trades = 0; wins = 0; losses = 0
    peak_r = Decimal(0)
    max_drawdown_r = Decimal(0)
    
    trade_log: List[Dict[str, Any]] = []
    
    all_pnl_r = []     
    gross_profit_r = Decimal(0) 
    gross_loss_r = Decimal(0)   
    
    scan_duration_ms = rates[1]['time'] - rates[0]['time']
    max_wait_ms = MAX_WAIT_CANDLES * scan_duration_ms
    
    skip_until_ts = 0

    # 6. BOUCLE PRINCIPALE
    for i in range(start_index, end_index):
        
        current_ts = rates[i]['time']
        
        if current_ts < skip_until_ts:
            continue
        
        # --- DÉTECTION DU SETUP (AVEC NOUVEAU FILTRE INTÉGRÉ) ---
        setup = detect_fvg_setup(rates, i, scale, stdev_threshold, stdev_max, allow_longs, allow_shorts)
        
        if setup:
            stop_loss_risk = abs(setup["entry_price"] - setup["sl_price"])
            if stop_loss_risk == 0: continue
            
            if setup["side"] == "LONG":
               target_price = setup["entry_price"] + stop_loss_risk * rr_ratio
            else: 
               target_price = setup["entry_price"] - stop_loss_risk * rr_ratio

            tp_price = qround(target_price, scale)
            
            simulation_start_ts = current_ts + scan_duration_ms
            expiration_ts = simulation_start_ts + max_wait_ms
            ltf_start_idx = bisect.bisect_left(ltf_timestamps, simulation_start_ts)
            
            if ltf_start_idx >= len(ltf_data):
                continue

            result, real_entry_ts, exit_ts = run_ltf_simulation_memory(
                ltf_data,             
                ltf_start_idx,        
                entry=float(setup["entry_price"]), 
                sl=float(setup["sl_price"]), 
                tp=float(tp_price), 
                side=setup["side"], 
                expiration_ts=expiration_ts
            )
            
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
                drawdown_r = peak_r - balance_r
                max_drawdown_r = max(max_drawdown_r, drawdown_r)
                
                trade_log.append({
                    "pair": pair, 
                    "entry_time": real_entry_ts if result != "EXPIRED" else rates[i]["time"], 
                    "exit_time": exit_ts,
                    "side": setup["side"], "entry_price": setup["entry_price"], "sl_price": setup["sl_price"],
                    "tp_price": tp_price, "exit_price": setup["sl_price"] if result == "LOSS" else tp_price,
                    "result": result, "pnl_r": pnl_r
                })
                
                skip_until_ts = exit_ts
    
    expectancy_r = balance_r / total_trades if total_trades > 0 else Decimal(0)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    max_drawdown_percent = max_drawdown_r * risk_per_trade * Decimal("100") 
    
    if gross_loss_r == 0:
        profit_factor = Decimal("99.99") 
    else:
        profit_factor = gross_profit_r / gross_loss_r
        
    sqn = 0.0
    if total_trades > 1:
        mean_pnl = statistics.mean(all_pnl_r)
        stdev_pnl = statistics.stdev(all_pnl_r)
        if stdev_pnl > 0:
            sqn = math.sqrt(total_trades) * (mean_pnl / stdev_pnl)

    GLOBAL_RESULTS.append({
        "pair": pair, "total_trades": total_trades, "wins": wins, "losses": losses,
        "expectancy_r": expectancy_r,
        "max_drawdown_r": max_drawdown_r,
        "max_drawdown_percent": max_drawdown_percent,
        "win_rate": win_rate,  
        "profit_factor": profit_factor, 
        "sqn": sqn,                     
        "net_profit_euro": 0,
        "final_balance": 0, 
        "start_ts": summary_start_ts, "end_ts": summary_end_ts
    })
    
    return trade_log


# ---------- FONCTIONS D'AFFICHAGE (Inchangées) ----------

def display_trade_details(all_trades: Dict[str, List[Dict[str, Any]]], show_all: bool = False):
    
    print("\n" + "="*110)
    if show_all:
        print("DÉTAILS DE TOUS LES TRADES EXÉCUTÉS")
    else:
        print("DÉTAILS DES 3 DERNIERS TRADES (Pour vérification manuelle)")
    print("="*110)

    for pair, log in all_trades.items():
        if not log: continue
            
        if show_all:
            trades_to_show = log
        else:
            trades_to_show = log[-3:]
        
        print(f"\n--- PAIRE: {pair} (Total: {len(log)} trades) ---")
        
        print("| {:<10} | {:<11} | {:<11} | {:<6} | {:>12} | {:>12} | {:>12} | {:>8} |".format(
            "RÉSULTAT", "ENTRY TIME", "EXIT TIME", "SIDE", "ENTRY", "SL", "TP", "PNL (R)"
        ))
        print("|" + "-"*12 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*8 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*10 + "|")

        for trade in trades_to_show:
            entry_time_str = format_ts(trade["entry_time"])
            exit_time_str = format_ts(trade["exit_time"])
            
            scale = 5 
            if "JPY" in pair or "XAU" in pair or "XAG" in pair: scale = 3 
            elif "USD" in pair and (len(pair) == 6 or "BTC" in pair or "BNB" in pair): scale = 2
            
            pnl_r_str = f"{float(trade['pnl_r']):+.2f}R"
            
            print("| {:<10} | {:<11} | {:<11} | {:<6} | {:>12.{s}f} | {:>12.{s}f} | {:>12.{s}f} | {:>8} |".format(
                trade["result"], entry_time_str, exit_time_str, trade["side"],
                float(trade["entry_price"]), float(trade["sl_price"]), float(trade["tp_price"]),
                pnl_r_str, s=scale
            ))
        print("-" * 110)

def display_summary_table(rr_ratio: Decimal, stdev_threshold: float, risk_perc: Decimal, stdev_max: float, results: List[Dict[str, Any]]):
    
    results.sort(key=lambda x: x['sqn'], reverse=True)
    
    total_trades_all = sum(res['total_trades'] for res in results)
    total_wins_all = sum(res['wins'] for res in results)
    total_losses_all = sum(res['losses'] for res in results)
    global_win_rate = (total_wins_all / total_trades_all) * 100 if total_trades_all > 0 else 0.0

    total_expectancy = sum(res['expectancy_r'] * res['total_trades'] for res in results if res['total_trades'] > 0)
    weighted_expectancy = total_expectancy / total_trades_all if total_trades_all > 0 else Decimal(0)
    
    
    print("\n" + "="*145)
    print(f"SUMMARY BACKTEST PANDAS EMA 200 + FIBO 61.8 + TREND FILTRE {TREND_FILTER_TF} (TF: {SCAN_TF}, RR: {rr_ratio}R, RISK: {risk_perc*Decimal(100)}%)")
    print("="*145)
    
    header = "| {:^10} | {:^6} | {:^8} | {:^10} | {:^10} | {:^10} | {:^10} | {:^8} | {:^8} |".format(
        "PAIRE", "TRADES", "WIN RATE", "EXPECTANCY", "PROFIT F.", "SQN", "MAX DD(R)", "GAINS", "PERTES"
    )
    separator = "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|"
    
    print(header)
    print(separator)
    
    for res in results:
        
        win_rate_str = f"{res['win_rate']:.2f}%"
        expectancy_r_str = f"{float(res['expectancy_r']):.4f}R"
        profit_factor_str = f"{float(res['profit_factor']):.2f}"
        sqn_str = f"{float(res['sqn']):.2f}"
        max_dd_r_str = f"{float(res['max_drawdown_r']):.2f}R"
        
        print("| {:<10} | {:>6} | {:>9} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} |".format(
            res['pair'], 
            res['total_trades'], 
            win_rate_str,
            expectancy_r_str,
            profit_factor_str,
            sqn_str,
            max_dd_r_str,
            res['wins'], 
            res['losses']
        ))
    
    print(separator)
    print("| {:<10} | {:>6} | {:>9.2f}% | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} |".format(
        "TOTAL", total_trades_all, global_win_rate, "", "", "", "", total_wins_all, total_losses_all
    ))

    print(separator)
    print(f"| {'TOTAL EXPECTANCY (Wgt Avg)':<56} | {'{0:.4f}R'.format(float(weighted_expectancy)):>12} | {'':<49} |")
    print("="*145 + "\n")


def display_hourly_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    hourly_stats = {h: {'wins': 0, 'losses': 0, 'total': 0} for h in range(24)}
    has_trades = False
    for pair, trades in all_trades_log.items():
        for trade in trades:
            has_trades = True
            entry_ts = trade['entry_time'] # ms
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
        wins = stats['wins']
        wr = (wins / total * 100) if total > 0 else 0.0
        if total > 0:
            print("| {:02d}:00  | {:^8} | {:^8} | {:>9.2f}% |".format(h, total, wins, wr))
        else:
            print("| {:02d}:00  | {:^8} | {:^8} | {:>9} |".format(h, "-", "-", "-"))
    print("="*50 + "\n")


def display_daily_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    daily_stats = {d: {'wins': 0, 'losses': 0, 'total': 0} for d in range(7)}
    day_names = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]
    has_trades = False
    for pair, trades in all_trades_log.items():
        for trade in trades:
            has_trades = True
            entry_ts = trade['entry_time']
            result = trade['result']
            dt = datetime.fromtimestamp(entry_ts / 1000, tz=UTC)
            day_idx = dt.weekday() # 0 = Lundi
            daily_stats[day_idx]['total'] += 1
            if result == "WIN": daily_stats[day_idx]['wins'] += 1
            elif result == "LOSS": daily_stats[day_idx]['losses'] += 1
    if not has_trades: return
    print("\n" + "="*50)
    print(" 📅 DAILY BREAKDOWN (UTC TIME)")
    print("="*50)
    print("| {:^10} | {:^8} | {:^8} | {:^10} |".format("DAY", "TRADES", "WINS", "WIN RATE"))
    print("|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|")
    for d in range(7):
        stats = daily_stats[d]
        total = stats['total']
        wins = stats['wins']
        wr = (wins / total * 100) if total > 0 else 0.0
        if total > 0:
            print("| {:<10} | {:^8} | {:^8} | {:>9.2f}% |".format(day_names[d], total, wins, wr))
        else:
            print("| {:<10} | {:^8} | {:^8} | {:>9} |".format(day_names[d], "-", "-", "-"))
    print("="*50 + "\n")


def display_side_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    long_stats = {'wins': 0, 'losses': 0, 'total': 0}
    short_stats = {'wins': 0, 'losses': 0, 'total': 0}
    has_trades = False
    
    for pair, trades in all_trades_log.items():
        for trade in trades:
            has_trades = True
            result = trade['result']
            side = trade['side']
            
            if side == "LONG":
                long_stats['total'] += 1
                if result == "WIN": long_stats['wins'] += 1
                elif result == "LOSS": long_stats['losses'] += 1
            elif side == "SHORT":
                short_stats['total'] += 1
                if result == "WIN": short_stats['wins'] += 1
                elif result == "LOSS": short_stats['losses'] += 1
                
    if not has_trades: return

    print("\n" + "="*60)
    print(" ⚖️ LONG / SHORT BREAKDOWN")
    print("="*60)
    print("| {:^10} | {:^8} | {:^8} | {:^8} | {:^10} |".format("SIDE", "TRADES", "WINS", "LOSSES", "WIN RATE"))
    print("|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|")
    
    # Calc Long WR
    l_wr = (long_stats['wins'] / long_stats['total'] * 100) if long_stats['total'] > 0 else 0.0
    print("| {:<10} | {:^8} | {:^8} | {:^8} | {:>9.2f}% |".format("LONG", long_stats['total'], long_stats['wins'], long_stats['losses'], l_wr))
    
    # Calc Short WR
    s_wr = (short_stats['wins'] / short_stats['total'] * 100) if short_stats['total'] > 0 else 0.0
    print("| {:<10} | {:^8} | {:^8} | {:^8} | {:>9.2f}% |".format("SHORT", short_stats['total'], short_stats['wins'], short_stats['losses'], s_wr))
    print("="*60 + "\n")


def get_asset_type(pair: str) -> str:
    p_up = pair.upper()
    if any(k in p_up for k in ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'LTC', 'BCH']): return "CRYPTO"
    if any(k in p_up for k in ['US30', 'SP500', 'NAS100', 'NSDQ', 'DOW', 'DAX']): return "INDICES"
    if any(k in p_up for k in ['XAU', 'XAG']): return "COMMODITIES"
    return "FOREX"

def display_keepers_csv(results: List[Dict[str, Any]]):
    print("\n" + "="*80)
    print(" 💎 KEEPER PAIRS (FILTRE: PF>=1.15 & SQN>=1.5)")
    print("="*80)
    keepers = [r for r in results if r['profit_factor'] >= Decimal("1.1") and r['sqn'] >= 1.1]
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
    max_drawdown_amount = Decimal(0)
    max_drawdown_pct = Decimal(0)
    current_r_balance = Decimal(0)
    high_water_mark_r = Decimal(0)
    max_drawdown_r = Decimal(0)
    total_net_profit = Decimal(0)

    for t in all_trades:
        risk_amount = current_balance * risk_per_trade
        pnl_currency = risk_amount * t['pnl_r']
        current_balance += pnl_currency
        total_net_profit += pnl_currency
        if current_balance > high_water_mark: high_water_mark = current_balance
        current_drawdown = high_water_mark - current_balance
        if current_drawdown > max_drawdown_amount: max_drawdown_amount = current_drawdown
        current_drawdown_pct = (current_drawdown / high_water_mark) * 100 if high_water_mark > 0 else 0
        if current_drawdown_pct > max_drawdown_pct: max_drawdown_pct = current_drawdown_pct
        current_r_balance += t['pnl_r']
        if current_r_balance > high_water_mark_r: high_water_mark_r = current_r_balance
        current_dd_r = high_water_mark_r - current_r_balance
        if current_dd_r > max_drawdown_r: max_drawdown_r = current_dd_r

    roi_pct = ((current_balance - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
    print("\n" + "="*60)
    print(" 💰 PORTFOLIO SIMULATION (ALL PAIRS CHRONOLOGICAL)")
    print("="*60)
    print(f"{'CAPITAL FINAL':<30} : {current_balance:,.2f} USD ({roi_pct:+.2f}%)")
    print(f"{'MAX DRAWDOWN (%)':<30} : -{max_drawdown_pct:.2f}%")
    print(f"{'MAX DRAWDOWN (R)':<30} : -{max_drawdown_r:.2f}R")
    print(f"{'TOTAL TRADES':<30} : {len(all_trades)}")
    print("="*60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Backtester EMA 200 / FIBO 61.8 Pandas.")
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--rr", type=Decimal, default=DEFAULT_RR)
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--end-date", type=str, default=None)
    ap.add_argument("--risk", type=Decimal, default=DEFAULT_RISK_PER_TRADE)
    ap.add_argument("--fees", type=Decimal, default=DEFAULT_FEES_PCT)
    # --- NOUVEAUX ARGUMENTS ---
    ap.add_argument("--no-long", action="store_true", help="Forcer la désactivation des trades LONG")
    ap.add_argument("--no-short", action="store_true", help="Forcer la désactivation des trades SHORT")
    
    args = ap.parse_args()
    
    # --- LOGIQUE CORRIGÉE ---
    allow_longs = DEFAULT_ALLOW_LONG and (not args.no_long)
    allow_shorts = DEFAULT_ALLOW_SHORT and (not args.no_short)
    
    start_ms = parse_date_to_ms(args.start_date) if args.start_date else None
    end_ms = parse_date_to_ms(args.end_date, is_end_date=True) if args.end_date else None
    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    if not pairs: sys.exit(1)
        
    print(f"Lancement du Backtest (TF: {SCAN_TF}, RR: {args.rr}, RISK: {args.risk * Decimal(100)}%)")
    print(f"DIRECTIONS AUTORISÉES -> LONG: {allow_longs} | SHORT: {allow_shorts}")
    print(f"FILTRE HTF ACTIVÉ SUR : {TREND_FILTER_TF}")
    
    all_trades_log: Dict[str, List[Dict[str, Any]]] = {} 

    for p in pairs:
        base, quote = p[:3], p[3:]
        scale = price_scale(base, quote)
        trade_log = execute_backtest(
            engine, p, args.rr, scale, 0.0, start_ms, end_ms, args.risk, 0.0, args.fees,
            allow_longs=allow_longs, allow_shorts=allow_shorts 
        )
        all_trades_log[p] = trade_log 
            
    display_trade_details(all_trades_log, show_all=SHOW_ALL_TRADES)
    display_summary_table(args.rr, 0.0, args.risk, 0.0, GLOBAL_RESULTS)
    display_hourly_breakdown(all_trades_log)
    display_daily_breakdown(all_trades_log)
    display_side_breakdown(all_trades_log)
    display_keepers_csv(GLOBAL_RESULTS)
    display_portfolio_simulation(all_trades_log, INITIAL_BALANCE, args.risk)

if __name__ == "__main__":
    main()