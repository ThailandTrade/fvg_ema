#!/usr/bin/env python3
# postgres_fvg_scanner_15m_v4.py
"""
PostgreSQL -> FVG/EMA50 Scanner (15M Only) - ADJUSTABLE FILTER
- Fetches data from PostgreSQL (table: candles_mt5_[pair]_[tf]).
- Checks for FVG, applies StDev filter, and verifies trend via EMA-50.
- Prints "Detected Setup" ONCE per candle per pair.
"""

import os, re, csv, sys, time
import argparse
import statistics
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List

# 🛑 Imports BDD basés sur VOTRE script d'ingestion
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, select, desc, inspect
from sqlalchemy.types import Numeric

UTC = timezone.utc

# ---------- CONFIG ----------
EMA_LEN_50 = 50
EMA_ALPHA_50 = Decimal("2") / Decimal(str(EMA_LEN_50 + 1))
STDEV_PERIOD = 200 # Période pour calculer la volatilité moyenne des gaps

SCAN_TF = "15m" # Timeframe utilisée dans le nom de la table PostgreSQL

# ---------- UTILS BDD & GENERALES ----------
def price_scale(base: str, quote: str) -> int:
    """Détermine le nombre de décimales pour un taux de change."""
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    """Arrondit un nombre avec le bon quantize."""
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    """Formate un timestamp (en SECONDES) en string UTC."""
    return datetime.fromtimestamp(ts, tz=UTC).strftime('%Y-%m-%d %H:%M')

def parse_pairs(path: str):
    """Charge la liste des paires à scanner depuis un fichier CSV."""
    out = []
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p:
                out.append(p.strip())
    return out

def sanitize_name(s: str) -> str:
    """Assainit le nom pour une utilisation comme nom de table PostgreSQL."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    """Crée et retourne le moteur de connexion PostgreSQL."""
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
        # Test de connexion rapide
        with engine.connect():
            pass 
        return engine
    except Exception as e:
        print(f"[FATAL] Échec de la connexion PostgreSQL. Vérifiez .env et le service : {e}")
        sys.exit(1)


# ---------- FONCTION DE RÉCUPÉRATION BDD CORRIGÉE ----------

def fetch_rates_from_db(engine, pair: str, tf: str, count: int) -> Optional[List[Dict[str, float]]]:
    """
    Récupère les 'count' dernières bougies fermées pour la paire/timeframe depuis PostgreSQL.
    """
    base, quote = pair[:3], pair[3:]
    scale = price_scale(base, quote)
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    # Définition de la structure minimale de la table
    meta = MetaData()
    table = Table(
        table_name, meta,
        Column("ts", BigInteger, primary_key=True),
        Column("high", Numeric(20, scale)),
        Column("low", Numeric(20, scale)),
        Column("close", Numeric(20, scale)),
    )
    
    # 🛑 CORRECTION DE L'ERREUR HAS_TABLE : Utilisation de inspect()
    inspector = inspect(engine) 
    if not inspector.has_table(table_name):
        # print(f"[WARN] Table {table_name} non trouvée dans la BDD.")
        return None

    try:
        with engine.connect() as conn:
            # Sélectionne les données dans l'ordre décroissant et limite à 'count'
            q = (
                select(table.c.ts.label("time"), table.c.high, table.c.low, table.c.close)
                .order_by(desc(table.c.ts))
                .limit(count)
            )
            rows = conn.execute(q).fetchall()
        
        if not rows:
            return None

        # Convertir les objets Row en dictionnaires et inverser l'ordre
        rates = []
        for row in reversed(rows): 
            rates.append({
                "time": int(row.time), # Timestamp en MS
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close)
            })

        return rates
    
    except Exception as e:
        # Affiche l'erreur de BDD si elle se produit (ex: colonne manquante)
        print(f"[ERR] BDD error for {pair}/{tf}: {e}")
        return None


# ---------- LOGIC (INCHANGÉE) ----------
def calculate_ema_series(rates, length=50) -> Optional[Decimal]:
    if len(rates) < length:
        return None
    
    closes = [Decimal(str(r["close"])) for r in rates]
    sma = sum(closes[:length]) / Decimal(length)
    ema = sma
    alpha = Decimal("2") / Decimal(str(length + 1))

    for c in closes[length:]:
        ema = alpha * c + (Decimal("1") - alpha) * ema
    
    return ema

def check_filtered_fvg(rates, threshold: float) -> Tuple[bool, bool, float]:
    """
    Checks for FVG and applies the StDev filter.
    Returns: (is_bullish, is_bearish, score)
    """
    if len(rates) < STDEV_PERIOD + 3:
        return False, False, 0.0

    c1 = rates[-3]["close"]
    h1 = rates[-3]["high"]
    l1 = rates[-3]["low"]
    
    c3 = rates[-1]["close"]
    h3 = rates[-1]["high"]
    l3 = rates[-1]["low"]

    raw_bull_cond = (h1 < l3) and (c3 > c1)
    raw_bear_cond = (l1 > h3) and (c3 < c1)

    if not raw_bull_cond and not raw_bear_cond:
        return False, False, 0.0

    # --- StDev Calculation (Optimized for speed) ---
    subset = rates[-(STDEV_PERIOD + 5):]
    lows = [r["low"] for r in subset]
    highs = [r["high"] for r in subset]
    
    diffs = []
    
    for i in range(2, len(lows)):
        # Calcule la taille absolue du gap général (volatilité)
        gap = abs(lows[i] - highs[i-2]) 
        diffs.append(gap)

    recent_diffs = diffs[-STDEV_PERIOD:]
    if len(recent_diffs) < 2: 
        return False, False, 0.0
        
    try:
        volatility = statistics.stdev(recent_diffs)
    except statistics.StatisticsError:
        volatility = 0.0
        
    if volatility == 0: volatility = 1.0e-5 # Avoid div/0

    # --- Apply Filter ---
    is_bullish = False
    is_bearish = False
    score = 0.0

    if raw_bull_cond:
        current_gap = l3 - h1
        score = current_gap / volatility
        if score > threshold:
            is_bullish = True

    elif raw_bear_cond:
        current_gap = l1 - h3
        score = current_gap / volatility
        if score > threshold:
            is_bearish = True

    return is_bullish, is_bearish, score

# ---------- FONCTION DE SCAN MODIFIÉE ----------

def scan_pair(engine, pair: str, last_alerts: Dict[str, int], threshold: float):
    
    # Récupération des données depuis la BDD (timestamp en MS)
    rates = fetch_rates_from_db(engine, pair, SCAN_TF, 450)

    if rates is None or len(rates) < 250:
        return

    last_candle_ts_ms = int(rates[-1]["time"]) 

    # Skip duplicate alerts
    if last_alerts.get(pair) == last_candle_ts_ms:
        return

    base, quote = pair[:3], pair[3:]
    scale = price_scale(base, quote)

    # 1. EMA 50
    ema_50 = calculate_ema_series(rates, EMA_LEN_50)
    if ema_50 is None:
        return

    # Utiliser le prix de clôture de la dernière bougie
    current_close = qround(rates[-1]["close"], scale)
    
    # 2. Filtered FVG Check
    is_bull_fvg, is_bear_fvg, score = check_filtered_fvg(rates, threshold)

    setup_msg = None

    # 3. Combine Logic (FVG + Tendance)
    if is_bull_fvg and current_close > ema_50:
        setup_msg = f"LONG (Score: {score:.2f})"
    elif is_bear_fvg and current_close < ema_50:
        setup_msg = f"SHORT (Score: {score:.2f})"

    # 4. Print
    if setup_msg:
        # Conversion MS -> Secondes pour format_ts
        ts_str = format_ts(last_candle_ts_ms // 1000) 
        print(f"Detected Setup : {pair} {setup_msg} | Bougie : {ts_str}")
        last_alerts[pair] = last_candle_ts_ms # Stocker le timestamp en MS


# ---------- MAIN FINAL ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    ap.add_argument("--threshold", type=float, default=0.2, help="Seuil de sensibilité FVG (Ex: 0.1=Très sensible, 0.5=Strict, 1.0=Massif uniquement)")
    args = ap.parse_args()

    # Initialisation du moteur PostgreSQL
    engine = get_pg_engine()

    pairs = parse_pairs(args.pairs_file)
    if not pairs:
        print("No pairs found in pairs.txt. Exiting.")
        sys.exit(1)

    print(f"Scanning {len(pairs)} pairs on {SCAN_TF} from PostgreSQL...")
    print(f"Filter Threshold: {args.threshold} (GapSize > {args.threshold} * StDev)")
    
    last_alerts = {}

    try:
        while True:
            for p in pairs:
                scan_pair(engine, p, last_alerts, args.threshold)
            
            # Attendre 10 secondes avant la prochaine boucle de scan
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[STOP] User interrupt. Scanner stopped.")
        pass

if __name__ == "__main__":
    main()