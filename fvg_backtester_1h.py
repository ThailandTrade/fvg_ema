#!/usr/bin/env python3
# postgres_fvg_backtester_r_based_v15_winrate_restored.py

import os, re, csv, sys, time
import argparse
import statistics
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List, Any

# Outils de Base de Données
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, select, desc, inspect, and_
from sqlalchemy.types import Numeric

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING ----------
DEFAULT_RR = Decimal("1.0") 
MAX_WAIT_CANDLES = 4 
SCAN_TF = "30m" # Unité de temps par défaut
INITIAL_BALANCE = Decimal("10000.00")
DEFAULT_RISK_PER_TRADE = Decimal("0.01") 

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


# --- Fetch Rates (Inchangée) ---

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
                conditions.append(table.c.ts >= (start_ms - STDEV_PERIOD * 60 * 15 * 1000)) 
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


# --- FVG Volatility Check & Detection (Inchangées) ---

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

def detect_fvg_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_threshold: float, stdev_max: float) -> Optional[Dict[str, Any]]:
    # Détection du setup FVG/EMA 50
    if i < 2: return None
    ema50 = rates[i]["ema_50"]; h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    if ema50 is None: return None
    is_bull_stdev, is_bear_stdev, score, current_gap = check_fvg_volatility(rates, i, stdev_threshold)
    if not is_bull_stdev and not is_bear_stdev: return None
    
    if score > stdev_max: # Filtre de borne haute du score
        return None

    ema_ok = False
    
    if is_bull_stdev:
        entry_side = "LONG"; fvg_high = Decimal(str(rates[i]["low"])); fvg_low = Decimal(str(h_i_2)); entry_price, sl_price = fvg_high, fvg_low
        ema_ok = fvg_high > Decimal(str(ema50))
    elif is_bear_stdev:
        entry_side = "SHORT"; fvg_high = Decimal(str(l_i_2)); fvg_low = Decimal(str(rates[i]["high"])); entry_price, sl_price = fvg_low, fvg_high
        ema_ok = fvg_low < Decimal(str(ema50))
    else: return None 

    if not ema_ok: return None
    
    return {
        "side": entry_side,
        "entry_price": qround(entry_price, scale), 
        "sl_price": qround(sl_price, scale),
        "fvg_start_candle_index": i,
        "stdev_score": score,
        "gap_size": current_gap
    }


# ---------- LOGIQUE DE BACKTESTING PRINCIPALE (Inchangée) ----------

def execute_backtest(rates: List[Dict[str, Any]], pair: str, rr_ratio: Decimal, scale: int, stdev_threshold: float, start_ms: Optional[int], end_ms: Optional[int], risk_per_trade: Decimal, stdev_max: float) -> List[Dict[str, Any]]:
    
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
    
    active_setup: Optional[Dict[str, Any]] = None
    trade_log: List[Dict[str, Any]] = []
    
    summary_start_ts = rates[start_index]['time']
    summary_end_ts = rates[end_index - 1]['time']
    
    # 2. Itération sur la plage filtrée
    for i in range(start_index, end_index):
        
        # --- 1. DÉTECTION DU SETUP FVG ---
        if active_setup is None:
            setup = detect_fvg_setup(rates, i, scale, stdev_threshold, stdev_max)
            if setup:
                active_setup = setup
                active_setup["wait_start_index"] = i 
                
                stop_loss_risk = abs(active_setup["entry_price"] - active_setup["sl_price"])
                
                if active_setup["side"] == "LONG":
                   target_price = active_setup["entry_price"] + stop_loss_risk * rr_ratio
                else: 
                   target_price = active_setup["entry_price"] - stop_loss_risk * rr_ratio

                active_setup["tp_price"] = qround(target_price, scale)
        
        # --- 2. GESTION DU TRADE EN ATTENTE / ENTREÉ ---
        elif active_setup is not None and not active_setup.get("is_open", False):
            
            current_wait_count = i - active_setup["wait_start_index"]
            if current_wait_count > MAX_WAIT_CANDLES:
                active_setup = None
                continue

            wait_candle = rates[i]; high = wait_candle["high"]; low = wait_candle["low"]; entry = float(active_setup["entry_price"])
            is_entered = False
            
            if active_setup["side"] == "LONG":
                if low <= entry: is_entered = True
            elif active_setup["side"] == "SHORT":
                if high >= entry: is_entered = True
            
            if is_entered:
                active_setup["is_open"] = True; active_setup["entry_index"] = i; total_trades += 1
                tp = float(active_setup["tp_price"]); sl = float(active_setup["sl_price"])
                trade_result = None
                
                # Vérification perte/gain sur la bougie d'entrée
                if active_setup["side"] == "LONG":
                    if low <= sl: trade_result = "LOSS"
                    elif high >= tp: trade_result = "WIN"
                elif active_setup["side"] == "SHORT":
                    if high >= sl: trade_result = "LOSS"
                    elif low <= tp: trade_result = "WIN"
                
                if trade_result:
                    pnl_r = rr_ratio if trade_result == "WIN" else Decimal("-1.0")
                    if trade_result == "WIN": wins += 1
                    else: losses += 1
                        
                    balance_r += pnl_r
                    
                    peak_r = max(peak_r, balance_r)
                    drawdown_r = peak_r - balance_r
                    max_drawdown_r = max(max_drawdown_r, drawdown_r)
                    
                    trade_log.append({
                        "pair": pair, "entry_time": rates[active_setup["entry_index"]]["time"], "exit_time": rates[i]["time"],
                        "side": active_setup["side"], "entry_price": active_setup["entry_price"], "sl_price": active_setup["sl_price"],
                        "tp_price": active_setup["tp_price"], "exit_price": active_setup["sl_price"] if trade_result == "LOSS" else active_setup["tp_price"],
                        "result": trade_result, "pnl_r": pnl_r
                    })
                    active_setup = None 
                
        # --- 3. GESTION DU TRADE OUVERT ---
        if active_setup is not None and active_setup.get("is_open", False) and i > active_setup["entry_index"]:
            
            check_candle = rates[i]; high = check_candle["high"]; low = check_candle["low"]
            tp = float(active_setup["tp_price"]); sl = float(active_setup["sl_price"])
            trade_result = None
            
            if active_setup["side"] == "LONG":
                if high >= tp: trade_result = "WIN"
                elif low <= sl: trade_result = "LOSS"
            elif active_setup["side"] == "SHORT":
                if low <= tp: trade_result = "WIN"
                elif high >= sl: trade_result = "LOSS"
            
            if trade_result:
                
                pnl_r = rr_ratio if trade_result == "WIN" else Decimal("-1.0")
                if trade_result == "WIN": wins += 1
                else: losses += 1
                        
                balance_r += pnl_r
                
                peak_r = max(peak_r, balance_r)
                drawdown_r = peak_r - balance_r
                max_drawdown_r = max(max_drawdown_r, drawdown_r)
                
                trade_log.append({
                    "pair": pair, "entry_time": rates[active_setup["entry_index"]]["time"], "exit_time": rates[i]["time"],
                    "side": active_setup["side"], "entry_price": active_setup["entry_price"], "sl_price": active_setup["sl_price"],
                    "tp_price": active_setup["tp_price"], "exit_price": active_setup["sl_price"] if trade_result == "LOSS" else active_setup["tp_price"],
                    "result": trade_result, "pnl_r": pnl_r
                })
                active_setup = None
    
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


# ---------- FONCTIONS D'AFFICHAGE (MODIFIÉES pour Totaux Globaux) ----------

def display_trade_details(all_trades: Dict[str, List[Dict[str, Any]]]):
    # Affiche les détails des derniers trades pour la vérification (non utilisé par défaut)
    
    print("\n" + "="*110)
    print("DÉTAILS DES 3 DERNIERS TRADES (Pour vérification manuelle)")
    print("="*110)

    for pair, log in all_trades.items():
        if not log: continue
            
        last_trades = log[-3:]
        
        print(f"\n--- PAIRE: {pair} (Derniers {len(last_trades)} trades) ---")
        
        print("| {:<10} | {:<11} | {:<11} | {:<6} | {:>12} | {:>12} | {:>12} | {:>8} |".format(
            "RÉSULTAT", "ENTRY TIME", "EXIT TIME", "SIDE", "ENTRY", "SL", "TP", "PNL (R)"
        ))
        print("|" + "-"*12 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*8 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*10 + "|")

        for trade in last_trades:
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
    
    # --- CALCULS DES TOTAUX GLOBAUX ---
    total_trades_all = sum(res['total_trades'] for res in results)
    total_wins_all = sum(res['wins'] for res in results)
    total_losses_all = sum(res['losses'] for res in results)
    
    # CALCUL DU WIN RATE GLOBAL
    global_win_rate = (total_wins_all / total_trades_all) * 100 if total_trades_all > 0 else 0.0
    
    # CALCUL DE L'EXPECTANCY PONDÉRÉE
    total_expectancy = sum(res['expectancy_r'] * res['total_trades'] for res in results if res['total_trades'] > 0)
    weighted_expectancy = total_expectancy / total_trades_all if total_trades_all > 0 else Decimal(0)
    
    
    print("\n" + "="*105)
    print(f"SUMMARY BACKTEST FVG/EMA 50 (TF: {SCAN_TF}, RR: {rr_ratio}R, RISK: {risk_perc*Decimal(100)}%)")
    print(f"STDEV FILTER: {stdev_threshold} < Score < {stdev_max}")
    print("="*105)
    
    # 🛑 TABLEAU PRINCIPAL (Win Rate inclus)
    
    header = "| {:^10} | {:^8} | {:^8} | {:^10} | {:^12} | {:^12} | {:^12} |".format(
        "PAIRE", "TRADES", "GAINS", "PERTES", "WIN RATE", "EXPECTANCY", "MAX DD (%)"
    )
    separator = "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|"
    
    print(header)
    print(separator)
    
    for res in results:
        
        win_rate_str = f"{res['win_rate']:.2f}%"
        expectancy_r_str = f"{float(res['expectancy_r']):.4f}R"
        max_dd_str = f"{float(res['max_drawdown_percent']):.2f}%"
        
        print("| {:<10} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12} | {:>12} |".format(
            res['pair'], 
            res['total_trades'], 
            res['wins'], 
            res['losses'], 
            win_rate_str,
            expectancy_r_str,
            max_dd_str
        ))
    
    # LIGNE DES TOTAUX GLOBALES (AJOUTÉE)
    total_win_rate_str = f"{global_win_rate:.2f}%"
    
    print(separator)
    print("| {:<10} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12} | {:>12} |".format(
        "TOTAL", 
        total_trades_all, 
        total_wins_all, 
        total_losses_all, 
        total_win_rate_str,
        "",
        ""
    ))
    
    print(separator)
    
    # Statistiques Globales (Expectancy Pondérée)
    print(f"| {'TOTAL EXPECTANCY (Wgt Avg)':<56} | {'{0:.4f}R'.format(float(weighted_expectancy)):>12} | {'':<12} |")
    print("="*105 + "\n")


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
            trade_log = execute_backtest(rates, p, args.rr, scale, args.stdev_threshold, start_ms, end_ms, args.risk, args.stdev_max)
            all_trades_log[p] = trade_log 

    # 2. Affichage du tableau récapitulatif
    display_summary_table(args.rr, args.stdev_threshold, args.risk, args.stdev_max, GLOBAL_RESULTS)
    
    # 3. Affichage du détail des derniers trades (commenté par défaut)
    # display_trade_details(all_trades_log)


if __name__ == "__main__":
    main()