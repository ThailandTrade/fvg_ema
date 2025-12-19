#!/usr/bin/env python3
# live_fvg_db_trader_30m_FIXED_HEADER_AUDIT_SOFT_EXP.py

import time
import sys
import os
import statistics
import MetaTrader5 as mt5
import re
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, List, Any

# BDD
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------- CONFIGURATION ----------
TIMEFRAME_STR = "30m"
TIMEFRAME_MT5_MIN = 30
MAGIC_NUMBER = 888888
ORDER_COMMENT = "FVG_Bot_Audit"

# Estimation des commissions (Ex: 5 USD par lot Round-Turn)
ESTIMATED_COMM_PER_LOT = 5.0 

# Paramètres Stratégie
MAX_WAIT_CANDLES = 4 
STDEV_PERIOD = 200
DEFAULT_RR = 1.0 
RISK_PERCENT = 0.003 # 0.1% par trade
STDEV_THRESHOLD = 0.5
STDEV_MAX = 1.0

# ---------- CONNEXION BDD ----------

def get_pg_engine():
    load_dotenv()
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "postgres")
    
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=disable"
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# ---------- RÉCUPÉRATION DONNÉES ----------

def fetch_last_candles_from_db(engine, pair: str, limit: int = 250) -> Optional[List[Dict[str, Any]]]:
    s_pair = sanitize_name(pair)
    s_tf = sanitize_name(TIMEFRAME_STR)
    table_name = f"candles_mt5_{s_pair}_{s_tf}"
    
    query = text(f"""
        SELECT ts, open, high, low, close, ema_50 
        FROM {table_name} 
        ORDER BY ts DESC 
        LIMIT :limit
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"limit": limit}).fetchall()
            
        if not result:
            return None
        
        if limit > 20 and len(result) < STDEV_PERIOD + 5:
            return None
            
        rows = list(reversed(result))
        
        rates = []
        for r in rows:
            rates.append({
                "time": int(r.ts),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "ema_50": float(r.ema_50) if r.ema_50 is not None else None
            })
            
        return rates
        
    except Exception as e:
        return None

# ---------- LOGIQUE DE DÉTECTION (50% / i-1) ----------

def check_fvg_volatility(rates: List[Dict[str, Any]], i: int, threshold: float) -> tuple[bool, bool, float, float]:
    if i < STDEV_PERIOD + 2: return False, False, 0.0, 0.0
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i = rates[i]["high"]; l_i = rates[i]["low"]
    
    raw_bull_cond = (h_i_2 < l_i)
    raw_bear_cond = (l_i_2 > h_i)
    
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
        current_gap = l_i - h_i_2
        score = current_gap / volatility
        if score > threshold: is_bullish = True
    elif raw_bear_cond:
        current_gap = l_i_2 - h_i
        score = current_gap / volatility
        if score > threshold: is_bearish = True
        
    return is_bullish, is_bearish, score, current_gap

def detect_setup_on_last_candle(rates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    i = len(rates) - 1
    ema50 = rates[i]["ema_50"]
    if ema50 is None: return None

    is_bull_stdev, is_bear_stdev, score, current_gap = check_fvg_volatility(rates, i, STDEV_THRESHOLD)
    if not is_bull_stdev and not is_bear_stdev: return None
    if score > STDEV_MAX: return None

    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i_1 = rates[i-1]["high"]; l_i_1 = rates[i-1]["low"] 
    h_i = rates[i]["high"];     l_i = rates[i]["low"]

    ema_ok = False
    entry_side = ""
    entry_price = 0.0
    sl_price = 0.0

    if is_bull_stdev:
        entry_side = "LONG"
        fvg_high = l_i; fvg_low = h_i_2
        entry_price = (fvg_high + fvg_low) / 2.0
        sl_price = l_i_1
        if entry_price > ema50 and sl_price < entry_price: ema_ok = True
        
    elif is_bear_stdev:
        entry_side = "SHORT"
        fvg_high = l_i_2; fvg_low = h_i
        entry_price = (fvg_high + fvg_low) / 2.0
        sl_price = h_i_1
        if entry_price < ema50 and sl_price > entry_price: ema_ok = True

    if not ema_ok: return None
    
    return {
        "side": entry_side,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "stdev_score": score,
        "ts": rates[i]["time"] 
    }

# ---------- EXÉCUTION MT5 & AUDIT ----------

def qround(x: float, scale: int) -> float:
    return float(Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP))

def get_lot_size(symbol: str, entry: float, sl: float, risk_percent: float, balance: float):
    distance = abs(entry - sl)
    if distance == 0: return 0.0
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info: return 0.0
    tick_value = symbol_info.trade_tick_value
    loss_per_lot = (distance / symbol_info.point) * tick_value
    if loss_per_lot == 0: return 0.0
    lots = (balance * risk_percent) / loss_per_lot
    step = symbol_info.volume_step
    lots = round(lots / step) * step
    return max(lots, symbol_info.volume_min)

# --- NOUVELLE FONCTION : GESTION EXPIRATION MANUELLE ---
# --- VERSION DEBUGGING ---
def check_and_clean_expired_orders(symbol: str):
    """
    Supprime les ordres trop vieux en se basant sur l'heure du SERVEUR (Tick),
    pour éviter les problèmes de fuseau horaire du PC.
    """
    orders = mt5.orders_get(symbol=symbol, magic=MAGIC_NUMBER)
    if not orders: return

    # 1. On récupère l'heure du serveur via le dernier tick
    last_tick = mt5.symbol_info_tick(symbol)
    if last_tick is None:
        # Si pas de tick (marché fermé/weekend), on ne peut pas fiablemont vérifier
        return
        
    server_time = last_tick.time # C'est un timestamp UNIX (int)
    
    # 2. Durée max
    max_duration_sec = TIMEFRAME_MT5_MIN * 60 * MAX_WAIT_CANDLES

    for order in orders:
        # On compare : Heure Serveur vs Heure Création Ordre (aussi Serveur)
        order_age_sec = server_time - order.time_setup
        order_age_min = int(order_age_sec / 60)
        
        # Debug pour vérifier que le chiffre n'est plus négatif
        # print(f"DEBUG {symbol}: Age {order_age_min} min (Serveur: {server_time} - Setup: {order.time_setup})")
        
        if order_age_sec > max_duration_sec:
            print(f"⌛ [EXPIRATION] Ordre {symbol} trop vieux ({order_age_min} min). Suppression...")
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "magic": MAGIC_NUMBER
            }
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                 print(f"⚠️ Echec suppression ordre {order.ticket}: {res.comment}")

def place_mt5_order(symbol: str, setup: Dict[str, Any]):
    # On fait un nettoyage préventif avant de placer un nouveau setup
    # (Bien que la logique principale le fasse, c'est une sécurité)
    check_and_clean_expired_orders(symbol)
    
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0: return

    orders = mt5.orders_get(symbol=symbol)
    # Si un ordre existe déjà et n'est pas expiré, on ne fait rien (ou on remplace selon la logique voulue)
    # Ici, on remplace s'il y en a un vieux
    if orders:
        for order in orders:
            request_delete = { "action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket, "magic": MAGIC_NUMBER }
            mt5.order_send(request_delete)
            print(f"♻️ [UPDATE] {symbol}: Ancien ordre supprimé pour mise à jour.")

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info: return
    account_info = mt5.account_info()
    if not account_info: return

    entry = setup['entry_price']; sl = setup['sl_price']
    dist_sl = abs(entry - sl)
    
    if setup['side'] == "LONG":
        tp = entry + (dist_sl * DEFAULT_RR)
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
    else:
        tp = entry - (dist_sl * DEFAULT_RR)
        order_type = mt5.ORDER_TYPE_SELL_LIMIT

    scale = symbol_info.digits
    entry = qround(entry, scale); sl = qround(sl, scale); tp = qround(tp, scale)
    lots = get_lot_size(symbol, entry, sl, RISK_PERCENT, account_info.balance)
    
    if lots == 0:
        print(f"❌ {symbol}: Impossible de calculer les lots (SL trop court ou erreur data).")
        return

    # --- CALCUL DES FRAIS (AUDIT) ---
    comm_cost_currency = lots * ESTIMATED_COMM_PER_LOT
    total_frais = comm_cost_currency
    risk_monetaire = account_info.balance * RISK_PERCENT
    ratio_frais = (total_frais / risk_monetaire) * 100 if risk_monetaire > 0 else 0

    print("\n" + "="*60)
    print(f"📢 SETUP DÉTECTÉ: {symbol} ({setup['side']})")
    print("-" * 60)
    print(f"   Entry : {entry}  |  SL : {sl}  |  TP : {tp}")
    print(f"   Risk  : {risk_monetaire:.2f} USD (approx)")
    print(f"   Lots  : {lots}")
    print("-" * 60)
    print(f"   📊 AUDIT FRAIS (COMMISSION ONLY) :")
    print(f"   Coût Comm.     : {comm_cost_currency:.2f} USD")
    print(f"   ---------------------------")
    warning = "🔴 DANGER (Frais > 20% Risk)" if ratio_frais > 20 else "🟢 OK"
    print(f"   IMPACT / RISK : {ratio_frais:.2f}%  {warning}")
    print("="*60 + "\n")

    # CORRECTION CRITIQUE : On passe en GTC (Good Till Cancelled)
    # car le broker refuse l'expiration via "ORDER_TIME_SPECIFIED".
    # L'expiration sera gérée par la fonction check_and_clean_expired_orders().
    
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": ORDER_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC, # <--- ICI : GTC pour éviter l'erreur 10022
        "expiration": 0,                 # <--- ICI : 0 requis pour GTC
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        t_str = datetime.fromtimestamp(setup['ts']/1000, tz=timezone.utc).strftime('%H:%M')
        print(f"✅ [EXEC] {symbol} {setup['side']} (Bougie {t_str} UTC) | Entry: {entry} | SL: {sl} | Lots: {lots}")
    else:
        print(f"[MT5 ERROR] {symbol}: {res.comment} ({res.retcode})")

# ---------- BOUCLE PRINCIPALE ----------

def main():
    if not mt5.initialize():
        print("[FATAL] MT5 Init failed")
        sys.exit(1)
    
    try:
        engine = get_pg_engine()
        print("✅ DB Connected.")
    except Exception as e:
        print(f"[FATAL] DB Connection failed: {e}")
        sys.exit(1)

    pairs = []
    files_to_try = ["session_pairs_e8markets.txt", "pairs.txt"]
    filename = ""
    for fn in files_to_try:
        if os.path.exists(fn):
            filename = fn
            break
    if not filename:
        print("❌ Aucun fichier de paires trouvé.")
        sys.exit(1)

    try:
        with open(filename, "r", newline="", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        for line in raw_lines:
            if "," in line:
                parts = line.split(",")
                p = parts[-1].strip()
            else:
                p = line.strip()
            if "pair" not in p.lower() and "type" not in p.lower():
                pairs.append(p)
    except Exception as e:
        print(f"Erreur lecture fichier paires: {e}")
        sys.exit(1)

    if not pairs:
        print(f"Aucune paire valide.")
        sys.exit(1)

    print(f"🚀 Bot LIVE 30m lancé | Risk: {RISK_PERCENT*100}% | Pairs: {len(pairs)}")
    print(f"Stratégie: Entry 50% FVG | SL i-1 | TF: {TIMEFRAME_STR}")
    print(f"Audit Frais: ACTIF (Comm: {ESTIMATED_COMM_PER_LOT} / lot)")
    print(f"⚠️ Expiration Gérée par Script (Le broker refuse l'expiration auto).")

    last_printed_ts = 0
    processed_setups = {}

    try:
        while True:
            # --- BLOC MONITEUR ---
            if pairs:
                ref_pair = pairs[0] 
                ref_rates = fetch_last_candles_from_db(engine, ref_pair, limit=5)
                if ref_rates:
                    last_db_ts = ref_rates[-1]['time']
                    if last_db_ts != last_printed_ts:
                        db_time_str = datetime.fromtimestamp(last_db_ts/1000, tz=timezone.utc).strftime('%H:%M')
                        print(f"\n--- 🕰️ DERNIÈRE DATA EN BASE ({ref_pair}) : {db_time_str} UTC ---")
                        last_printed_ts = last_db_ts
            # ---------------------
            
            for pair in pairs:
                # 1. NETTOYAGE DES ORDRES EXPIRÉS (SOFT EXPIRATION)
                check_and_clean_expired_orders(pair)
                
                # 2. DETECTION DE NOUVEAUX SETUPS
                rates = fetch_last_candles_from_db(engine, pair, limit=STDEV_PERIOD + 10)
                if not rates: continue
                
                setup = detect_setup_on_last_candle(rates)
                
                if setup:
                    last_processed_ts = processed_setups.get(pair)
                    current_setup_ts = setup['ts']
                    
                    if last_processed_ts != current_setup_ts:
                        place_mt5_order(pair, setup)
                        processed_setups[pair] = current_setup_ts
            
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nArrêt.")
        mt5.shutdown()

if __name__ == "__main__":
    main()