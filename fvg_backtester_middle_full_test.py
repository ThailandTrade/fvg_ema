#!/usr/bin/env python3
# postgres_fvg_backtester_PANDAS_SMART_FULL.py

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
DEFAULT_RR = Decimal("0.9")
MAX_WAIT_CANDLES = 5
SCAN_TF = "30m"           # Unité de temps pour la détection du setup
EXECUTION_TF_SUFFIX = "3m" # Timeframe pour l'exécution précise
DEFAULT_RISK_PER_TRADE = Decimal("0.003") 
DEFAULT_FEES_PCT = Decimal("0.10") # 10% de frais par trade (calculé sur le risque 1R)

# --- CONFIG SMART MONEY FILTERS (NOUVEAU) ---
# Filtre Displacement : La bougie doit être X fois plus grande que la moyenne
USE_DISPLACEMENT_FILTER = True
MIN_BODY_RATIO = 1.2 

# Filtre Momentum : RSI > 50 pour Long, RSI < 50 pour Short
USE_RSI_FILTER = True
RSI_PERIOD = 14

# --- PARAMETRES DU SUMMARY ---
INITIAL_BALANCE = Decimal("50000.00") # Capital de départ pour la simulation
SHOW_ALL_TRADES = False               # Mettre à True pour voir la liste complète, False pour les 3 derniers

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


# --- Fetch Rates (OPTIMISÉ AVEC PANDAS + CALCUL INDICATEURS) ---

def calculate_rsi_pandas(series, period=14):
    """Calcul du RSI vectorisé avec Pandas"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Fill NaN avec 50 (neutre) au début

def fetch_htf_data_pandas(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    """
    Charge les données HTF (30m) via Pandas et pré-calcule :
    1. StDev (Volatilité)
    2. RSI (Momentum)
    3. Body Size Average (Displacement)
    """
    base, quote = pair[:3], pair[3:]
    scale = price_scale(base, quote)
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    # Construction de la requête SQL brute
    query = f"SELECT ts as time, open, high, low, close, ema_50 FROM {table_name}"
    conditions = []
    
    # Buffer de sécurité pour le calcul StDev/RSI
    safe_buffer_ms = timedelta(days=20).total_seconds() * 1000
    
    if start_ms is not None:
        conditions.append(f"ts >= {start_ms - safe_buffer_ms}")
    if end_ms is not None:
        conditions.append(f"ts <= {end_ms}")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY ts ASC"

    try:
        # Chargement ultra-rapide en DataFrame
        df = pd.read_sql(query, engine)
        if df.empty:
            return None

        # --- PRÉ-CALCUL DU STDEV (VECTORISÉ) ---
        gap_series = (df['low'] - df['high'].shift(2)).abs()
        df['stdev_200'] = gap_series.rolling(window=STDEV_PERIOD).std(ddof=1)
        df['stdev_200'] = df['stdev_200'].fillna(0.0)

        # --- PRÉ-CALCUL DU RSI (VECTORISÉ) ---
        if USE_RSI_FILTER:
            df['rsi'] = calculate_rsi_pandas(df['close'], RSI_PERIOD)
        else:
            df['rsi'] = 50.0 # Neutre si désactivé

        # --- PRÉ-CALCUL DU BODY SIZE (VECTORISÉ pour Displacement) ---
        if USE_DISPLACEMENT_FILTER:
            # Taille du corps absolue
            df['body_size'] = (df['close'] - df['open']).abs()
            # Moyenne des 10 dernières bougies (shift(1) pour ne pas inclure la bougie actuelle dans sa propre moyenne)
            df['avg_body_10'] = df['body_size'].rolling(window=10).mean().shift(1).fillna(0.0)
        else:
            df['body_size'] = 0.0
            df['avg_body_10'] = 0.0

        return df.to_dict('records')

    except Exception as e:
        print(f"[ERR] Pandas Fetch HTF Error for {pair}/{tf}: {e}")
        return None


# --- SIMULATION LTF (OPTIMISÉE: CHARGEMENT EN BLOC) ---

def fetch_ltf_data_pandas(engine, pair: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    """
    Charge TOUTES les données LTF nécessaires pour la période en une seule fois.
    Renvoie une liste de dictionnaires pour itération rapide.
    """
    table_ltf = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    
    # On ajoute une petite marge de fin pour couvrir le dernier trade potentiel
    buffer_end = end_ms + (MAX_WAIT_CANDLES * 30 * 60 * 1000 * 2) # Large buffer
    
    query = f"SELECT ts, high, low FROM {table_ltf} WHERE ts >= {start_ms} AND ts <= {buffer_end} ORDER BY ts ASC"
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return []
        
        # On renvoie une liste de dicts, c'est le plus léger à itérer
        return df.to_dict('records')
        
    except Exception as e:
        # print(f"[WARN] LTF Data missing for {pair}: {e}")
        return []

def run_ltf_simulation_memory(ltf_data: List[Dict[str, Any]], start_index: int, entry: float, sl: float, tp: float, side: str, expiration_ts: int) -> Tuple[str, int]:
    """
    Simule le trade en parcourant la liste ltf_data déjà chargée en mémoire.
    Plus aucune connexion BDD ici !
    """
    is_open = False
    
    # On parcourt la liste à partir de l'index trouvé par bisect
    # On met une limite de sécurité (ex: 5000 bougies max à vérifier) pour éviter boucle infinie si bug data
    max_steps = 5000 
    
    # Itération protégée pour ne pas dépasser la taille de la liste
    end_loop = min(start_index + max_steps, len(ltf_data))
    
    for i in range(start_index, end_loop):
        row = ltf_data[i]
        ts = row['ts'] # Déjà int
        high = float(row['high'])
        low = float(row['low'])
        
        # 1. TENTATIVE D'ENTRÉE (PENDING)
        if not is_open:
            if ts > expiration_ts:
                return "EXPIRED", ts
            
            if side == "LONG":
                if low <= entry:
                    is_open = True
                    if low <= sl: return "LOSS", ts # SL touché sur la minute d'entrée
            elif side == "SHORT":
                if high >= entry:
                    is_open = True
                    if high >= sl: return "LOSS", ts # SL touché sur la minute d'entrée
        
        # 2. GESTION DU TRADE (OPEN)
        if is_open:
            if side == "LONG":
                if low <= sl: return "LOSS", ts
                if high >= tp: return "WIN", ts
            elif side == "SHORT":
                if high >= sl: return "LOSS", ts
                if low <= tp: return "WIN", ts
                
    # Si on sort de la boucle (fin des données), on considère expiré
    if ltf_data:
        last_ts = ltf_data[end_loop - 1]['ts']
    else:
        last_ts = expiration_ts
        
    return "EXPIRED", last_ts


# --- FVG Volatility Check & Detection (MODIFIÉE) ---

def check_fvg_volatility_optimized(rates: List[Dict[str, Any]], i: int, threshold: float) -> Tuple[bool, bool, float, float]:
    # Vérifie si le FVG est suffisamment grand par rapport à la volatilité (StDev PRÉ-CALCULÉ)
    if i < STDEV_PERIOD + 2: return False, False, 0.0, 0.0
    
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i = rates[i]["high"]; l_i = rates[i]["low"]
    
    raw_bull_cond = (h_i_2 < l_i)
    raw_bear_cond = (l_i_2 > h_i)
    
    if not raw_bull_cond and not raw_bear_cond: return False, False, 0.0, 0.0
    
    # ICI : On récupère le StDev pré-calculé par Pandas
    volatility = rates[i].get('stdev_200', 0.0)
    
    if volatility == 0: volatility = 1.0e-9 
    
    is_bullish = False; is_bearish = False; score = 0.0; current_gap = 0.0
    
    if raw_bull_cond:
        current_gap = l_i - h_i_2; score = current_gap / volatility
        if score > threshold: is_bullish = True
    elif raw_bear_cond:
        current_gap = l_i_2 - h_i; score = current_gap / volatility
        if score > threshold: is_bearish = True
        
    return is_bullish, is_bearish, score, current_gap

# ----------------------------------------------------------------------
# 🛑 FONCTION ENTRÉE : 50% FVG, SL sur bougie i-1, AVEC SMART FILTERS
# ----------------------------------------------------------------------

def detect_fvg_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_threshold: float, stdev_max: float) -> Optional[Dict[str, Any]]:
    # Détection du setup FVG/EMA 50
    if i < 20: return None # Besoin d'un peu plus d'historique pour les moyennes
    
    # Extraction des données des trois bougies
    ema50 = rates[i]["ema_50"]
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i_1 = rates[i-1]["high"]; l_i_1 = rates[i-1]["low"] # Données de la bougie i-1
    
    # Données pour les filtres avancés
    current_rsi = rates[i]["rsi"]
    current_body = rates[i]["body_size"]
    avg_body = rates[i]["avg_body_10"]
    
    if ema50 is None or pd.isna(ema50): return None
    
    # Appel version optimisée (qui lit le stdev pré-calculé)
    is_bull_stdev, is_bear_stdev, score, current_gap = check_fvg_volatility_optimized(rates, i, stdev_threshold)
    if not is_bull_stdev and not is_bear_stdev: return None
    
    if score > stdev_max: # Filtre de borne haute du score
        return None

    # --- FILTRE 1 : DISPLACEMENT (SMC) ---
    # Si la bougie qui crée le FVG n'est pas significativement plus grande que la moyenne, on filtre.
    if USE_DISPLACEMENT_FILTER and avg_body > 0:
        ratio = current_body / avg_body
        if ratio < MIN_BODY_RATIO:
            return None # Rejet : Manque de puissance (Displacement)

    ema_ok = False
    rsi_ok = False
    entry_price = Decimal(0)
    sl_price = Decimal(0)
    
    # --- LOGIQUE D'ENTRÉE ET SL ---
    
    if is_bull_stdev:
        entry_side = "LONG"
        
        # FVG (Gap) est entre l_i et h_i_2
        fvg_high = Decimal(str(rates[i]["low"]))   # l_i (Haut du FVG)
        fvg_low = Decimal(str(h_i_2))              # h_i_2 (Bas du FVG)
        
        # 1. Nouvelle Entrée: 50% du FVG (Midpoint)
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        
        # 2. SL ORIGINE: Low de la bougie i-1 (l_i_1)
        sl_price = Decimal(str(l_i_1))
        
        # Filtre EMA 50: L'entrée doit être au-dessus de l'EMA 50
        ema_ok = entry_price > Decimal(str(ema50))
        
        # --- FILTRE 2 : MOMENTUM REGIME (RSI) ---
        # En tendance haussière saine, le RSI doit soutenir le mouvement (> 50)
        if USE_RSI_FILTER:
            rsi_ok = current_rsi > 50
        else:
            rsi_ok = True
        
        # Validation du SL: SL doit être strictement inférieur à l'entrée (Long)
        if sl_price >= entry_price:
             return None

    elif is_bear_stdev:
        entry_side = "SHORT"
        
        # FVG (Gap) est entre l_i_2 et h_i
        fvg_high = Decimal(str(l_i_2))             # l_i_2 (Haut du FVG)
        fvg_low = Decimal(str(rates[i]["high"]))   # h_i (Bas du FVG)
        
        # 1. Nouvelle Entrée: 50% du FVG (Midpoint)
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        
        # 2. SL ORIGINE: High de la bougie i-1 (h_i_1)
        sl_price = Decimal(str(h_i_1))
        
        # Filtre EMA 50: L'entrée doit être en-dessous de l'EMA 50
        ema_ok = entry_price < Decimal(str(ema50))
        
        # --- FILTRE 2 : MOMENTUM REGIME (RSI) ---
        # En tendance baissière saine, le RSI doit soutenir le mouvement (< 50)
        if USE_RSI_FILTER:
            rsi_ok = current_rsi < 50
        else:
            rsi_ok = True
        
        # Validation du SL: SL doit être strictement supérieur à l'entrée (Short)
        if sl_price <= entry_price:
             return None
             
    else: return None  

    if not ema_ok: return None
    if not rsi_ok: return None # Rejet par le filtre RSI
    
    # L'entrée est conditionnée au prix 'entry_price' calculé
    return {
        "side": entry_side,
        "entry_price": qround(entry_price, scale),  
        "sl_price": qround(sl_price, scale),
        "fvg_start_candle_index": i,
        "stdev_score": score,
        "gap_size": current_gap
    }
# ----------------------------------------------------------------------


# ---------- LOGIQUE DE BACKTESTING PRINCIPALE (OPTIMISÉ + SQN/PF) ----------

def execute_backtest(engine, pair: str, rr_ratio: Decimal, scale: int, stdev_threshold: float, start_ms: Optional[int], end_ms: Optional[int], risk_per_trade: Decimal, stdev_max: float, fees_pct: Decimal) -> List[Dict[str, Any]]:
    
    # 1. CHARGEMENT HTF (30m) avec PANDAS
    rates = fetch_htf_data_pandas(engine, pair, SCAN_TF, start_ms, end_ms)
    
    if not rates or len(rates) < 200: return []
    
    # Détermination des bornes temporelles réelles des données chargées
    data_start_ms = rates[0]['time']
    data_end_ms = rates[-1]['time']
    
    # 2. CHARGEMENT LTF (3m) EN UNE FOIS (Gros gain de perf)
    # On charge tout ce qui couvre la période HTF + un buffer
    ltf_data = fetch_ltf_data_pandas(engine, pair, data_start_ms, data_end_ms)
    
    # Création d'une liste de timestamps LTF pour la recherche rapide (bisect)
    ltf_timestamps = [r['ts'] for r in ltf_data]
    
    # 3. FILTRAGE DE LA PLAGE DE SCAN (Start/End dates utilisateur)
    start_index = 0
    end_index = len(rates)
    required_seed = STDEV_PERIOD + 2 
    
    if start_ms is not None:
        for idx in range(required_seed, len(rates)):
            if rates[idx]['time'] >= start_ms:
                start_index = idx
                break
        else:
            return []

    if end_ms is not None:
        for idx in range(start_index, len(rates)):
            if rates[idx]['time'] > end_ms:
                end_index = idx
                break

    if end_index <= start_index: return []
    
    # --- CORRECTIF POUR BUG NAME ERROR ---
    summary_start_ts = rates[start_index]['time']
    summary_end_ts = rates[end_index - 1]['time']
    # -------------------------------------
    
    # Initialisation R-BASED
    balance_r = Decimal(0)
    total_trades = 0; wins = 0; losses = 0
    peak_r = Decimal(0)
    max_drawdown_r = Decimal(0)
    
    trade_log: List[Dict[str, Any]] = []
    
    # --- NOUVELLES METRIQUES: SQN & PF ---
    all_pnl_r = []     # Pour SQN
    gross_profit_r = Decimal(0) # Pour PF
    gross_loss_r = Decimal(0)   # Pour PF
    
    # Calcul de la durée d'une bougie Scan en ms (pour expiration et lookahead)
    scan_duration_ms = 0
    if len(rates) > 1:
        scan_duration_ms = rates[1]['time'] - rates[0]['time']
    if scan_duration_ms == 0: scan_duration_ms = 300000 # Default 5m if error
    
    max_wait_ms = MAX_WAIT_CANDLES * scan_duration_ms
    
    # Variable pour sauter le scan tant qu'un trade est en cours
    skip_until_ts = 0

    # 4. BOUCLE PRINCIPALE (Itération sur la plage filtrée)
    for i in range(start_index, end_index):
        
        current_ts = rates[i]['time']
        
        # Si un trade est en cours sur le LTF, on saute les bougies scan
        if current_ts < skip_until_ts:
            continue
        
        # --- 1. DÉTECTION DU SETUP FVG ---
        setup = detect_fvg_setup(rates, i, scale, stdev_threshold, stdev_max)
        
        if setup:
            stop_loss_risk = abs(setup["entry_price"] - setup["sl_price"])
            
            if setup["side"] == "LONG":
               target_price = setup["entry_price"] + stop_loss_risk * rr_ratio
            else: 
               target_price = setup["entry_price"] - stop_loss_risk * rr_ratio

            tp_price = qround(target_price, scale)
            
            # --- 2. EXECUTION LTF AVEC FIX LOOK-AHEAD ---
            # Le signal est clôturé à la fin de la bougie 'i'.
            # L'exécution commence donc au début de la bougie 'i+1'.
            
            simulation_start_ts = current_ts + scan_duration_ms
            expiration_ts = simulation_start_ts + max_wait_ms
            
            # --- OPTIMISATION: RECHERCHE DICHOTOMIQUE (BISECT) ---
            # On cherche l'index dans ltf_data où ts >= simulation_start_ts
            ltf_start_idx = bisect.bisect_left(ltf_timestamps, simulation_start_ts)
            
            # Sécurité: si l'index est hors limites (pas de data LTF pour cette période)
            if ltf_start_idx >= len(ltf_data):
                continue

            # --- EXECUTION EN MÉMOIRE ---
            result, exit_ts = run_ltf_simulation_memory(
                ltf_data,            # On passe la LISTE COMPLÈTE
                ltf_start_idx,       # On donne le POINT DE DÉPART
                entry=float(setup["entry_price"]), 
                sl=float(setup["sl_price"]), 
                tp=float(tp_price), 
                side=setup["side"], 
                expiration_ts=expiration_ts
            )
            
            # --- 3. TRAITEMENT DU RÉSULTAT AVEC FRAIS ---
            if result in ["WIN", "LOSS"]:
                total_trades += 1
                
                # --- CALCUL PNL AVEC FRAIS ---
                # Frais = fees_pct * 1R
                # Si WIN: +RR - fees
                # Si LOSS: -1R - fees
                
                if result == "WIN":
                    pnl_r = rr_ratio - fees_pct
                    wins += 1
                    gross_profit_r += pnl_r # Pour Profit Factor
                else:
                    pnl_r = Decimal("-1.0") - fees_pct
                    losses += 1
                    gross_loss_r += abs(pnl_r) # On somme les pertes en absolu
                
                # Stockage pour SQN
                all_pnl_r.append(float(pnl_r))
                
                balance_r += pnl_r
                peak_r = max(peak_r, balance_r)
                drawdown_r = peak_r - balance_r
                max_drawdown_r = max(max_drawdown_r, drawdown_r)
                
                trade_log.append({
                    "pair": pair, "entry_time": rates[i]["time"], "exit_time": exit_ts,
                    "side": setup["side"], "entry_price": setup["entry_price"], "sl_price": setup["sl_price"],
                    "tp_price": tp_price, "exit_price": setup["sl_price"] if result == "LOSS" else tp_price,
                    "result": result, "pnl_r": pnl_r
                })
                
                # On ne prend pas de nouveau trade tant que celui-ci n'est pas fini
                skip_until_ts = exit_ts
    
    # --- Collecte des résultats pour le tableau final ---
    
    expectancy_r = balance_r / total_trades if total_trades > 0 else Decimal(0)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    max_drawdown_percent = max_drawdown_r * risk_per_trade * Decimal("100") 
    
    # --- CALCUL FINAL PROFIT FACTOR ---
    if gross_loss_r == 0:
        profit_factor = Decimal("99.99") # Infini
    else:
        profit_factor = gross_profit_r / gross_loss_r
        
    # --- CALCUL FINAL SQN ---
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
        "profit_factor": profit_factor, # Nouveau
        "sqn": sqn,                     # Nouveau
        "net_profit_euro": 0,
        "final_balance": 0, 
        "start_ts": summary_start_ts, "end_ts": summary_end_ts
    })
    
    return trade_log


# ---------- FONCTIONS D'AFFICHAGE ----------

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
    
    # TRI PAR SQN (La robustesse d'abord !)
    results.sort(key=lambda x: x['sqn'], reverse=True)
    
    # AJOUT TOTAL
    total_trades_all = sum(res['total_trades'] for res in results)
    total_wins_all = sum(res['wins'] for res in results)
    total_losses_all = sum(res['losses'] for res in results)
    global_win_rate = (total_wins_all / total_trades_all) * 100 if total_trades_all > 0 else 0.0

    total_expectancy = sum(res['expectancy_r'] * res['total_trades'] for res in results if res['total_trades'] > 0)
    weighted_expectancy = total_expectancy / total_trades_all if total_trades_all > 0 else Decimal(0)
    
    
    print("\n" + "="*145)
    print(f"SUMMARY BACKTEST PANDAS FVG/EMA 50 (TF: {SCAN_TF}, RR: {rr_ratio}R, RISK: {risk_perc*Decimal(100)}%, FEES: {DEFAULT_FEES_PCT*100}%)")
    print(f"STDEV FILTER: {stdev_threshold} < Score < {stdev_max}")
    print(f"EXECUTION TF: {EXECUTION_TF_SUFFIX}")
    print("="*145)
    
    # TABLEAU AVEC SQN ET PROFIT FACTOR
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
    
    # AFFICHAGE DU TOTAL
    print(separator)
    # CORRECTIF: on passe le float directement ou on formate avant
    print("| {:<10} | {:>6} | {:>9.2f}% | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} |".format(
        "TOTAL", total_trades_all, global_win_rate, "", "", "", "", total_wins_all, total_losses_all
    ))

    print(separator)
    
    # Statistiques Globales
    print(f"| {'TOTAL EXPECTANCY (Wgt Avg)':<56} | {'{0:.4f}R'.format(float(weighted_expectancy)):>12} | {'':<49} |")
    print("="*145 + "\n")


# --- BREAKDOWN PAR HEURE ---

def display_hourly_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    """
    Affiche un tableau agrégé des performances par heure (UTC).
    """
    hourly_stats = {h: {'wins': 0, 'losses': 0, 'total': 0} for h in range(24)}
    
    has_trades = False
    
    for pair, trades in all_trades_log.items():
        for trade in trades:
            has_trades = True
            entry_ts = trade['entry_time'] # ms
            result = trade['result']
            
            # Conversion Timestamp -> Heure UTC
            dt = datetime.fromtimestamp(entry_ts / 1000, tz=UTC)
            hour = dt.hour
            
            hourly_stats[hour]['total'] += 1
            if result == "WIN":
                hourly_stats[hour]['wins'] += 1
            elif result == "LOSS":
                hourly_stats[hour]['losses'] += 1
    
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


# --- BREAKDOWN PAR JOUR DE LA SEMAINE ---

def display_daily_breakdown(all_trades_log: Dict[str, List[Dict[str, Any]]]):
    """
    Affiche un tableau agrégé des performances par jour de la semaine.
    Lundi (0) -> Dimanche (6)
    """
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
            if result == "WIN":
                daily_stats[day_idx]['wins'] += 1
            elif result == "LOSS":
                daily_stats[day_idx]['losses'] += 1
    
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


# --- AJOUT: FONCTIONS D'AFFICHAGE CSV POUR KEEPER ---

def get_asset_type(pair: str) -> str:
    """Détermine le type d'actif en fonction du nom de la paire."""
    p_up = pair.upper()
    
    if any(k in p_up for k in ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'LTC', 'BCH', 'DOGE', 'DOT', 'LINK', 'MATIC', 'UNI', 'AVAX', 'TRX']):
        return "CRYPTO"
    if any(k in p_up for k in ['US30', 'SP500', 'SPX', 'NAS100', 'NSDQ', 'NDX', 'DOW', 'DAX', 'GER30', 'GER40', 'CAC', 'UK100', 'FTSE', 'JP225', 'NIKKEI', 'ASX', 'IBEX', 'STOXX']):
        return "INDICES"
    if any(k in p_up for k in ['XAU', 'XAG', 'WTI', 'BRENT', 'OIL', 'NATGAS', 'COPPER']):
        return "COMMODITIES"
        
    return "FOREX"

def display_keepers_csv(results: List[Dict[str, Any]]):
    """Affiche les paires à conserver AVEC FILTRE INTELLIGENT."""
    
    print("\n" + "="*80)
    print(" 💎 KEEPER PAIRS (FILTRE: Trades>=30 & ProfitFactor>=1.3 & WR>=50%)")
    print("="*80)

    # LOGIQUE DE FILTRE AMÉLIORÉE ICI
    keepers = []
    for r in results:
        if (r['profit_factor'] >= 1.0 and 
            r['win_rate'] >= 60.0):
            keepers.append(r)
            
    # Tri par SQN pour avoir les meilleurs en premier
    keepers.sort(key=lambda x: x['sqn'], reverse=True)

    print("type,pair")
    for res in keepers:
        pair_name = res['pair']
        asset_type = get_asset_type(pair_name)
        print(f"{asset_type},{pair_name}")
    print("\n")


# --- SIMULATION GLOBALE DU CAPITAL (PORTFOLIO) ---

def display_portfolio_simulation(all_trades_log: Dict[str, List[Dict[str, Any]]], initial_capital: Decimal, risk_per_trade: Decimal):
    """
    Agrège tous les trades de toutes les paires, les trie chronologiquement par date de sortie,
    et simule l'évolution du capital.
    Ajout: Calcul du Max Drawdown en R.
    """
    # 1. Aplatir le dictionnaire en une seule liste
    all_trades = []
    for pair, trades in all_trades_log.items():
        for t in trades:
            # On s'assure que la paire est dans l'objet trade (au cas où ce n'est pas le cas dans le log d'origine)
            trade_copy = t.copy()
            trade_copy['pair'] = pair
            all_trades.append(trade_copy)
            
    if not all_trades:
        print("\n[INFO] Aucun trade pour simuler le portefeuille.")
        return

    # 2. Trier par date de sortie (EXIT TIME) pour simuler la séquence réelle
    all_trades.sort(key=lambda x: x['exit_time'])

    current_balance = initial_capital
    high_water_mark = initial_capital
    max_drawdown_amount = Decimal(0)
    max_drawdown_pct = Decimal(0)
    
    # Vars pour Drawdown en R
    current_r_balance = Decimal(0)
    high_water_mark_r = Decimal(0)
    max_drawdown_r = Decimal(0)
    
    total_net_profit = Decimal(0)

    # 3. Simulation trade par trade
    for t in all_trades:
        # Risque en devise = Balance Actuelle * % Risque
        risk_amount = current_balance * risk_per_trade
        
        # PnL en devise = Montant Risqué * R généré (déjà impacté par les frais)
        pnl_currency = risk_amount * t['pnl_r']
        
        # --- Gestion Capital ---
        current_balance += pnl_currency
        total_net_profit += pnl_currency
        
        if current_balance > high_water_mark:
            high_water_mark = current_balance
        
        current_drawdown = high_water_mark - current_balance
        if current_drawdown > max_drawdown_amount:
            max_drawdown_amount = current_drawdown
            
        current_drawdown_pct = (current_drawdown / high_water_mark) * 100 if high_water_mark > 0 else 0
        if current_drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = current_drawdown_pct
            
        # --- Gestion R (Drawdown en R) ---
        current_r_balance += t['pnl_r']
        
        if current_r_balance > high_water_mark_r:
            high_water_mark_r = current_r_balance
            
        current_dd_r = high_water_mark_r - current_r_balance
        if current_dd_r > max_drawdown_r:
            max_drawdown_r = current_dd_r

    # 4. Affichage
    roi_pct = ((current_balance - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
    
    print("\n" + "="*60)
    print(" 💰 PORTFOLIO SIMULATION (ALL PAIRS CHRONOLOGICAL)")
    print("="*60)
    print(f"{'CAPITAL INITIAL':<30} : {initial_capital:,.2f}")
    print(f"{'CAPITAL FINAL':<30} : {current_balance:,.2f}")
    print(f"{'NET PROFIT':<30} : {total_net_profit:,.2f} ({roi_pct:+.2f}%)")
    print("-" * 60)
    print(f"{'MAX DRAWDOWN (Montant)':<30} : -{max_drawdown_amount:,.2f}")
    print(f"{'MAX DRAWDOWN (%)':<30} : -{max_drawdown_pct:.2f}%")
    print(f"{'MAX DRAWDOWN (R)':<30} : -{max_drawdown_r:.2f}R")
    print(f"{'TOTAL TRADES':<30} : {len(all_trades)}")
    print("="*60 + "\n")


# ---------- MAIN (Inchangée) ----------
def main():
    ap = argparse.ArgumentParser(description="Backtester FVG/EMA 50 Optimisé Pandas.")
    ap.add_argument("--pairs-file", default="pairs.txt", help="Fichier CSV listant les paires à tester.")
    ap.add_argument("--rr", type=Decimal, default=DEFAULT_RR, help=f"Ratio Risk/Reward (défaut: {DEFAULT_RR}).")
    ap.add_argument("--stdev-threshold", type=float, default=DEFAULT_STDEV_THRESHOLD, help=f"Seuil FVG/StDev Minimum (défaut: {DEFAULT_STDEV_THRESHOLD}).")
    ap.add_argument("--stdev-max", type=float, default=DEFAULT_STDEV_MAX, help=f"Borne haute FVG/StDev Maximum (défaut: {DEFAULT_STDEV_MAX}).")
    ap.add_argument("--start-date", type=str, default=None, help="Date de début du backtest (format: YYYY-MM-DD).")
    ap.add_argument("--end-date", type=str, default=None, help="Date de fin du backtest (format: YYYY-MM-DD).")
    
    ap.add_argument("--risk", type=Decimal, default=DEFAULT_RISK_PER_TRADE, help=f"Risque par trade (en décimal, ex: 0.02 pour 2%.) Défaut: {DEFAULT_RISK_PER_TRADE}.")
    
    # NOUVEL ARGUMENT POUR LES FRAIS
    ap.add_argument("--fees", type=Decimal, default=DEFAULT_FEES_PCT, help=f"Pourcentage de frais par trade en décimal (ex: 0.10 pour 10%). Défaut: {DEFAULT_FEES_PCT}")

    args = ap.parse_args()
    
    start_ms = None
    if args.start_date:
        try: start_ms = parse_date_to_ms(args.start_date)
        except ValueError as e: print(f"[ERROR] Date de début non valide: {e}"); sys.exit(1)

    end_ms = None
    if args.end_date:
        try: end_ms = parse_date_to_ms(args.end_date, is_end_date=True) 
        except ValueError as e: print(f"[ERROR] Date de fin non valide: {e}"); sys.exit(1)
            
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        print("[ERROR] La date de début doit être strictement antérieure à la date de fin.")
        sys.exit(1)

    if args.risk <= 0 or args.risk >= 1:
        print("[ERROR] Le risque par trade doit être une valeur décimale entre 0 et 1 (ex: 0.01 pour 1%).")
        sys.exit(1)
        
    if args.stdev_threshold >= args.stdev_max:
        print("[ERROR] Le Seuil StDev Minimum doit être inférieur à la Borne Haute StDev Maximum.")
        sys.exit(1)

    engine = get_pg_engine()

    pairs = parse_pairs(args.pairs_file)
    if not pairs:
        print("No pairs found in pairs.txt. Exiting.")
        sys.exit(1)
        
    print(f"Lancement du Backtest PANDAS (TF: {SCAN_TF}, RR: {args.rr}, RISK: {args.risk * Decimal(100)}%) sur {len(pairs)} paires.")
    print(f"Frais appliqués: {args.fees * 100}% par trade (Impact sur R).")
    
    if start_ms or end_ms:
        start_str = format_ts(start_ms) if start_ms else "Début des données"
        end_str = format_ts(end_ms) if end_ms else "Fin des données"
        print(f"Période de Backtest: {start_str} -> {end_str}")
    
    all_trades_log: Dict[str, List[Dict[str, Any]]] = {} 

    # 1. Exécution du backtest pour chaque paire
    for p in pairs:
        base, quote = p[:3], p[3:]
        scale = price_scale(base, quote)
        
        # Appel de la version Pandas
        trade_log = execute_backtest(
            engine, p, args.rr, scale, args.stdev_threshold, 
            start_ms, end_ms, args.risk, args.stdev_max, 
            args.fees 
        )
        all_trades_log[p] = trade_log 
            
    # 2. Affichage des détails des trades (Utilise la constante SHOW_ALL_TRADES)
    #display_trade_details(all_trades_log, show_all=SHOW_ALL_TRADES)

    # 3. Affichage du tableau récapitulatif par paire (AVEC SQN & PF)
    display_summary_table(args.rr, args.stdev_threshold, args.risk, args.stdev_max, GLOBAL_RESULTS)
    
    # 4. Affichage du breakdown par HEURE
    #display_hourly_breakdown(all_trades_log)

    # 5. Affichage du breakdown par JOUR
    display_daily_breakdown(all_trades_log)

    # 6. Affichage CSV des keepers (AVEC FILTRE INTELLIGENT)
    display_keepers_csv(GLOBAL_RESULTS)
    
    # 7. Affichage de la simulation de portefeuille globale (Utilise la constante INITIAL_BALANCE)
    display_portfolio_simulation(all_trades_log, INITIAL_BALANCE, args.risk)

if __name__ == "__main__":
    main()