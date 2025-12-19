#!/usr/bin/env python3
# postgres_fvg_backtester_FULL_BREAKDOWN.py

import os, re, csv, sys, time
import argparse
import statistics
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List, Any

# Outils de Base de Données
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, select, desc, inspect, and_, text
from sqlalchemy.types import Numeric

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING (PARAMETRES ICI) ----------
DEFAULT_RR = Decimal("1.0")
MAX_WAIT_CANDLES = 4
SCAN_TF = "30m"          # Unité de temps par défaut
EXECUTION_TF_SUFFIX = "1m" # Timeframe pour l'exécution précise
DEFAULT_RISK_PER_TRADE = Decimal("0.003") 

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


# --- Fetch Rates (CORRIGÉ POUR HISTORIQUE) ---

def fetch_all_rates(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    # Récupère les données des bougies
    base, quote = pair[:3], pair[3:]
    scale = price_scale(base, quote)
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    meta = MetaData()
    table = Table(
        table_name, meta,
        Column("ts", BigInteger, primary_key=True),
        Column("open", Numeric(20, scale)),
        Column("high", Numeric(20, scale)),
        Column("low", Numeric(20, scale)),
        Column("close", Numeric(20, scale)),
        Column("ema_50", Numeric(20, scale)), 
    )
    
    inspector = inspect(engine) 
    if not inspector.has_table(table_name):
        return None

    try:
        with engine.connect() as conn:
            
            conditions = []
            if start_ms is not None:
                # Ajout de l'historique nécessaire au calcul du STDEV
                # CORRECTION APPLIQUÉE : Buffer de 20 jours pour éviter les erreurs de calcul STDEV
                safe_buffer_ms = timedelta(days=20).total_seconds() * 1000
                conditions.append(table.c.ts >= (start_ms - safe_buffer_ms)) 
            if end_ms is not None:
                conditions.append(table.c.ts <= end_ms)
            
            q = select(
                table.c.ts.label("time"), 
                table.c.open, 
                table.c.high, 
                table.c.low, 
                table.c.close,
                table.c.ema_50
            ).where(and_(*conditions)).order_by(table.c.ts.asc())
            
            rows = conn.execute(q).fetchall()
        
        if not rows:
            return None

        rates = []
        for row in rows: 
            rates.append({
                "time": int(row.time), 
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "ema_50": float(row.ema_50) if row.ema_50 is not None else None
            })

        return rates
        
    except Exception as e:
        print(f"[ERR] BDD error for {pair}/{tf}: {e}")
        return None


# --- SIMULATION M1 (NOUVELLE FONCTION) ---

def run_m1_simulation(engine, pair: str, start_ts: int, entry: float, sl: float, tp: float, side: str, expiration_ts: int) -> Tuple[str, int]:
    """
    Simule l'entrée et la sortie sur des données M1.
    Retourne (Resultat, TimestampFin).
    """
    table_m1 = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    limit_candles = 50000 # On charge un bloc suffisant pour couvrir le trade
    
    sql = text(f"SELECT ts, high, low FROM {table_m1} WHERE ts >= :start_ts ORDER BY ts ASC LIMIT :limit")
    
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"start_ts": start_ts, "limit": limit_candles}).fetchall()
            
        if not rows:
            return "EXPIRED", start_ts
            
        is_open = False
        
        for row in rows:
            ts = int(row.ts)
            high = float(row.high)
            low = float(row.low)
            
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
        
        return "EXPIRED", rows[-1].ts 
        
    except Exception as e:
        # print(f"[WARN] M1 Simulation Error: {e}")
        return "ERROR", start_ts


# --- FVG Volatility Check & Detection (MODIFIÉE) ---

def check_fvg_volatility(rates: List[Dict[str, Any]], i: int, threshold: float) -> Tuple[bool, bool, float, float]:
    # Vérifie si le FVG est suffisamment grand par rapport à la volatilité (StDev)
    if i < STDEV_PERIOD + 2: return False, False, 0.0, 0.0
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i = rates[i]["high"]; l_i = rates[i]["low"]
    raw_bull_cond = (h_i_2 < l_i); raw_bear_cond = (l_i_2 > h_i)
    if not raw_bull_cond and not raw_bear_cond: return False, False, 0.0, 0.0
    subset = rates[i - STDEV_PERIOD - 2: i + 1]
    lows = [r["low"] for r in subset]; highs = [r["high"] for r in subset]
    diffs = []
    for k in range(2, len(lows)):
        gap = abs(lows[k] - highs[k-2])
        diffs.append(gap)
    recent_diffs = diffs[-STDEV_PERIOD:]
    try:
        volatility = statistics.stdev(recent_diffs)
    except statistics.StatisticsError: return False, False, 0.0, 0.0
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
# 🛑 FONCTION MODIFIÉE : Entrée au 50% FVG, SL sur bougie i-1 
# ----------------------------------------------------------------------

def detect_fvg_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_threshold: float, stdev_max: float) -> Optional[Dict[str, Any]]:
    # Détection du setup FVG/EMA 50
    if i < 2: return None
    
    # Extraction des données des trois bougies
    ema50 = rates[i]["ema_50"]
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i_1 = rates[i-1]["high"]; l_i_1 = rates[i-1]["low"] # Données de la bougie i-1
    
    if ema50 is None: return None
    
    is_bull_stdev, is_bear_stdev, score, current_gap = check_fvg_volatility(rates, i, stdev_threshold)
    if not is_bull_stdev and not is_bear_stdev: return None
    
    if score > stdev_max: # Filtre de borne haute du score
        return None

    ema_ok = False
    entry_price = Decimal(0)
    sl_price = Decimal(0)
    
    # --- LOGIQUE D'ENTRÉE ET SL MODIFIÉE (50% FVG & SL sur i-1) ---
    
    if is_bull_stdev:
        entry_side = "LONG"
        
        # FVG (Gap) est entre l_i et h_i_2
        fvg_high = Decimal(str(rates[i]["low"]))   # l_i (Haut du FVG)
        fvg_low = Decimal(str(h_i_2))              # h_i_2 (Bas du FVG)
        
        # 1. Nouvelle Entrée: 50% du FVG (Midpoint)
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        
        # 2. Nouveau SL: Low de la bougie i-1 (l_i_1)
        sl_price = Decimal(str(l_i_1))
        
        # Filtre EMA 50: L'entrée doit être au-dessus de l'EMA 50
        ema_ok = entry_price > Decimal(str(ema50))
        
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
        
        # 2. Nouveau SL: High de la bougie i-1 (h_i_1)
        sl_price = Decimal(str(h_i_1))
        
        # Filtre EMA 50: L'entrée doit être en-dessous de l'EMA 50
        ema_ok = entry_price < Decimal(str(ema50))
        
        # Validation du SL: SL doit être strictement supérieur à l'entrée (Short)
        if sl_price <= entry_price:
             return None
             
    else: return None  

    if not ema_ok: return None
    
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


# ---------- LOGIQUE DE BACKTESTING PRINCIPALE (Inchangée) ----------

def execute_backtest(engine, rates: List[Dict[str, Any]], pair: str, rr_ratio: Decimal, scale: int, stdev_threshold: float, start_ms: Optional[int], end_ms: Optional[int], risk_per_trade: Decimal, stdev_max: float) -> List[Dict[str, Any]]:
    
    if len(rates) < 200: return []
    
    # 1. Déterminer la plage d'itération réelle
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
    
    # Initialisation R-BASED
    balance_r = Decimal(0)
    total_trades = 0; wins = 0; losses = 0
    peak_r = Decimal(0)
    max_drawdown_r = Decimal(0)
    
    trade_log: List[Dict[str, Any]] = []
    
    summary_start_ts = rates[start_index]['time']
    summary_end_ts = rates[end_index - 1]['time']
    
    # Calcul de la durée d'une bougie Scan en ms (pour expiration et lookahead)
    scan_duration_ms = 0
    if len(rates) > 1:
        scan_duration_ms = rates[1]['time'] - rates[0]['time']
    if scan_duration_ms == 0: scan_duration_ms = 300000 # Default 5m if error
    
    max_wait_ms = MAX_WAIT_CANDLES * scan_duration_ms
    
    # Variable pour sauter le scan tant qu'un trade est en cours
    skip_until_ts = 0

    # 2. Itération sur la plage filtrée
    for i in range(start_index, end_index):
        
        current_ts = rates[i]['time']
        
        # Si un trade est en cours sur le M1, on saute les bougies scan
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
            
            # --- 2. EXECUTION M1 AVEC FIX LOOK-AHEAD ---
            # Le signal est clôturé à la fin de la bougie 'i'.
            # L'exécution commence donc au début de la bougie 'i+1'.
            # Timestamp i+1 = current_ts + scan_duration_ms
            
            simulation_start_ts = current_ts + scan_duration_ms
            expiration_ts = simulation_start_ts + max_wait_ms
            
            # Sécurité: ne pas simuler dans le futur si pas de données
            if simulation_start_ts > rates[-1]['time']:
                continue

            result, exit_ts = run_m1_simulation(
                engine, pair, 
                start_ts=simulation_start_ts, 
                entry=float(setup["entry_price"]), 
                sl=float(setup["sl_price"]), 
                tp=float(tp_price), 
                side=setup["side"], 
                expiration_ts=expiration_ts
            )
            
            # --- 3. TRAITEMENT DU RÉSULTAT ---
            if result in ["WIN", "LOSS"]:
                total_trades += 1
                pnl_r = rr_ratio if result == "WIN" else Decimal("-1.0")
                if result == "WIN": wins += 1
                else: losses += 1
                
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
    
    GLOBAL_RESULTS.append({
        "pair": pair, "total_trades": total_trades, "wins": wins, "losses": losses,
        "expectancy_r": expectancy_r,
        "max_drawdown_r": max_drawdown_r,
        "max_drawdown_percent": max_drawdown_percent,
        "win_rate": win_rate,  
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
    
    # TRI PAR EXPECTANCY R DÉCROISSANTE
    results.sort(key=lambda x: x['expectancy_r'], reverse=True)
    
    # AJOUT TOTAL
    total_trades_all = sum(res['total_trades'] for res in results)
    total_wins_all = sum(res['wins'] for res in results)
    total_losses_all = sum(res['losses'] for res in results)
    global_win_rate = (total_wins_all / total_trades_all) * 100 if total_trades_all > 0 else 0.0

    total_expectancy = sum(res['expectancy_r'] * res['total_trades'] for res in results if res['total_trades'] > 0)
    weighted_expectancy = total_expectancy / total_trades_all if total_trades_all > 0 else Decimal(0)
    
    
    print("\n" + "="*120)
    print(f"SUMMARY BACKTEST FVG/EMA 50 (TF: {SCAN_TF}, RR: {rr_ratio}R, RISK: {risk_perc*Decimal(100)}%)")
    print(f"STDEV FILTER: {stdev_threshold} < Score < {stdev_max}")
    print("="*120)
    
    # 🛑 TABLEAU PRINCIPAL (Avec MAX DD en R)
    
    header = "| {:^10} | {:^8} | {:^8} | {:^8} | {:^10} | {:^12} | {:^12} | {:^12} |".format(
        "PAIRE", "TRADES", "GAINS", "PERTES", "WIN RATE", "EXPECTANCY", "MAX DD (R)", "MAX DD (%)"
    )
    separator = "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|"
    
    print(header)
    print(separator)
    
    for res in results:
        
        win_rate_str = f"{res['win_rate']:.2f}%"
        expectancy_r_str = f"{float(res['expectancy_r']):.4f}R"
        max_dd_r_str = f"{float(res['max_drawdown_r']):.2f}R"
        max_dd_pct_str = f"{float(res['max_drawdown_percent']):.2f}%"
        
        print("| {:<10} | {:>8} | {:>8} | {:>8} | {:>10} | {:>12} | {:>12} | {:>12} |".format(
            res['pair'], 
            res['total_trades'], 
            res['wins'], 
            res['losses'], 
            win_rate_str,
            expectancy_r_str,
            max_dd_r_str,
            max_dd_pct_str
        ))
    
    # AFFICHAGE DU TOTAL
    print(separator)
    print("| {:<10} | {:>8} | {:>8} | {:>8} | {:>10} | {:>12} | {:>12} | {:>12} |".format(
        "TOTAL", total_trades_all, total_wins_all, total_losses_all, f"{global_win_rate:.2f}%", "", "", ""
    ))

    print(separator)
    
    # Statistiques Globales
    print(f"| {'TOTAL EXPECTANCY (Wgt Avg)':<56} | {'{0:.4f}R'.format(float(weighted_expectancy)):>12} | {'':<27} |")
    print("="*120 + "\n")


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


# --- NOUVEAU: BREAKDOWN PAR JOUR DE LA SEMAINE ---

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


# --- AJOUT: FONCTIONS D'AFFICHAGE CSV POUR KEEPER > 60% WR ---

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
    """Affiche les paires à conserver (> 60% WR) au format CSV pour copy-paste."""
    keepers = [r for r in results if r['win_rate'] > 60.0]
    
    print("type,pair")
    
    for res in keepers:
        pair_name = res['pair']
        asset_type = get_asset_type(pair_name)
        print(f"{asset_type},{pair_name}")


# --- NOUVEAU: SIMULATION GLOBALE DU CAPITAL (PORTFOLIO) ---

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
        
        # PnL en devise = Montant Risqué * R généré
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
    ap = argparse.ArgumentParser(description="Backtester FVG/EMA 50 basé sur PostgreSQL.")
    ap.add_argument("--pairs-file", default="pairs.txt", help="Fichier CSV listant les paires à tester.")
    ap.add_argument("--rr", type=Decimal, default=DEFAULT_RR, help=f"Ratio Risk/Reward (défaut: {DEFAULT_RR}).")
    ap.add_argument("--stdev-threshold", type=float, default=DEFAULT_STDEV_THRESHOLD, help=f"Seuil FVG/StDev Minimum (défaut: {DEFAULT_STDEV_THRESHOLD}).")
    ap.add_argument("--stdev-max", type=float, default=DEFAULT_STDEV_MAX, help=f"Borne haute FVG/StDev Maximum (défaut: {DEFAULT_STDEV_MAX}).")
    ap.add_argument("--start-date", type=str, default=None, help="Date de début du backtest (format: YYYY-MM-DD).")
    ap.add_argument("--end-date", type=str, default=None, help="Date de fin du backtest (format: YYYY-MM-DD).")
    
    ap.add_argument("--risk", type=Decimal, default=DEFAULT_RISK_PER_TRADE, help=f"Risque par trade (en décimal, ex: 0.02 pour 2%.) Défaut: {DEFAULT_RISK_PER_TRADE}.")
    
    # NOTE: Plus d'arguments CLI pour le capital ou l'affichage des trades.
    # Modifiez les variables INITIAL_BALANCE et SHOW_ALL_TRADES en haut du script.

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
        
    print(f"Lancement du Backtest FVG/EMA 50 (TF: {SCAN_TF}, RR: {args.rr}, RISK: {args.risk * Decimal(100)}%) sur {len(pairs)} paires.")
    if start_ms or end_ms:
        start_str = format_ts(start_ms) if start_ms else "Début des données"
        end_str = format_ts(end_ms) if end_ms else "Fin des données"
        print(f"Période de Backtest: {start_str} -> {end_str}")
    
    all_trades_log: Dict[str, List[Dict[str, Any]]] = {} 

    # 1. Exécution du backtest pour chaque paire
    for p in pairs:
        base, quote = p[:3], p[3:]
        scale = price_scale(base, quote)
        
        rates = fetch_all_rates(engine, p, SCAN_TF, start_ms, end_ms)
        
        if rates:
            # Modif: on passe 'engine' pour la simulation M1
            trade_log = execute_backtest(engine, rates, p, args.rr, scale, args.stdev_threshold, start_ms, end_ms, args.risk, args.stdev_max)
            all_trades_log[p] = trade_log 
            
    # 2. Affichage des détails des trades (Utilise la constante SHOW_ALL_TRADES)
    display_trade_details(all_trades_log, show_all=SHOW_ALL_TRADES)

    # 3. Affichage du tableau récapitulatif par paire
    display_summary_table(args.rr, args.stdev_threshold, args.risk, args.stdev_max, GLOBAL_RESULTS)
    
    # 4. Affichage du breakdown par HEURE
    display_hourly_breakdown(all_trades_log)

    # 5. Affichage du breakdown par JOUR
    display_daily_breakdown(all_trades_log)

    # 6. Affichage CSV des keepers
    #display_keepers_csv(GLOBAL_RESULTS)
    
    # 7. Affichage de la simulation de portefeuille globale (Utilise la constante INITIAL_BALANCE)
    display_portfolio_simulation(all_trades_log, INITIAL_BALANCE, args.risk)

if __name__ == "__main__":
    main()