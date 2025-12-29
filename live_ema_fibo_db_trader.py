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
TIMEFRAME_STR = "5m"           # Adapté pour la nouvelle stratégie (M5)
TIMEFRAME_MT5_MIN = 5
MAGIC_NUMBER = 888888
ORDER_COMMENT = "GoldenTrend"

# Estimation des commissions (Ex: 5 USD par lot Round-Turn)
ESTIMATED_COMM_PER_LOT = 5.0 

# Paramètres Stratégie Golden Trend
MAX_WAIT_CANDLES = 32         # Expiration
EMA_TREND_PERIOD = 200
FIB_RETREACEMENT = 0.618
SWING_CONFIRMATION_LAG = 5 
STDEV_PERIOD = 200            # Gardé pour compatibilité structurelle

DEFAULT_RR = 2.0              # Mis à jour pour la nouvelle stratégie
RISK_PERCENT = 0.0005         # 0.05% par trade (Exemple)
STDEV_THRESHOLD = 0.5
STDEV_MAX = 1.0

# --- FILTRES DE DIRECTION (NOUVEAU) ---
ALLOW_LONGS = True    # Mettre à False pour interdire les achats
ALLOW_SHORTS = False   # Mettre à False pour interdire les ventes

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
    
def calculate_ema_pandas_style(prices: List[float], period: int) -> float:
    if len(prices) < period: return 0.0
    alpha = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price * alpha) + (ema * (1 - alpha))
    return ema

# ---------- RÉCUPÉRATION DONNÉES ----------

def fetch_last_candles_from_db(engine, pair: str, limit: int = 250) -> Optional[List[Dict[str, Any]]]:
    s_pair = sanitize_name(pair)
    s_tf = sanitize_name(TIMEFRAME_STR)
    table_name = f"candles_mt5_{s_pair}_{s_tf}"
    
    query = text(f"""
        SELECT ts, open, high, low, close
        FROM {table_name} 
        ORDER BY ts DESC 
        LIMIT :limit
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"limit": limit}).fetchall()
            
        if not result:
            return None
        
        rows = list(reversed(result))
        
        rates = []
        for r in rows:
            rates.append({
                "time": int(r.ts),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close)
            })
            
        return rates
        
    except Exception as e:
        return None

# ---------- LOGIQUE DE DÉTECTION (EMA 200 / FIBO 61.8) ----------

def identify_swings(rates: List[Dict[str, Any]], lag: int) -> tuple[List[bool], List[bool]]:
    is_sh = [False] * len(rates)
    is_sl = [False] * len(rates)
    for i in range(lag, len(rates) - lag):
        if rates[i]['high'] == max(r['high'] for r in rates[i-lag : i+lag+1]):
            is_sh[i] = True
        if rates[i]['low'] == min(r['low'] for r in rates[i-lag : i+lag+1]):
            is_sl[i] = True
    return is_sh, is_sl

def detect_setup_on_last_candle(rates: List[Dict[str, Any]], allow_longs: bool, allow_shorts: bool) -> Optional[Dict[str, Any]]:
    if len(rates) < EMA_TREND_PERIOD + 20: return None
    
    # Calcul EMA 200 (Brain - Corrigé pour matcher Pandas)
    closes = [r['close'] for r in rates]
    ema_200 = calculate_ema_pandas_style(closes, EMA_TREND_PERIOD)
    
    curr = rates[-1]
    is_uptrend = curr['close'] > ema_200
    is_downtrend = curr['close'] < ema_200
    
    # Identification Swings avec Lag (Vision)
    is_sh, is_sl = identify_swings(rates, SWING_CONFIRMATION_LAG)
    vision_limit = len(rates) - 1 - SWING_CONFIRMATION_LAG
    
    entry_side = ""
    entry_price = 0.0
    sl_price = 0.0

    # --- LOGIQUE LONG (Si autorisé) ---
    if allow_longs and is_uptrend:
        sh_idx = -1
        for k in range(vision_limit, vision_limit - 60, -1):
            if is_sh[k]: sh_idx = k; break
        if sh_idx == -1: return None
        sl_idx = -1
        for k in range(sh_idx - 1, sh_idx - 100, -1):
            if is_sl[k]: sl_idx = k; break
        if sl_idx == -1: return None
        
        high_p = rates[sh_idx]['high']
        low_p = rates[sl_idx]['low']
        fib_lvl = high_p - ((high_p - low_p) * FIB_RETREACEMENT)
        if fib_lvl > ema_200 and curr['close'] > fib_lvl:
            entry_side = "LONG"
            entry_price = fib_lvl
            sl_price = low_p

    # --- LOGIQUE SHORT (Si autorisé) ---
    elif allow_shorts and is_downtrend:
        sl_idx = -1
        for k in range(vision_limit, vision_limit - 60, -1):
            if is_sl[k]: sl_idx = k; break
        if sl_idx == -1: return None
        sh_idx = -1
        for k in range(sl_idx - 1, sl_idx - 100, -1):
            if is_sh[k]: sh_idx = k; break
        if sh_idx == -1: return None
        
        low_p = rates[sl_idx]['low']
        high_p = rates[sh_idx]['high']
        fib_lvl = low_p + ((high_p - low_p) * FIB_RETREACEMENT)
        if fib_lvl < ema_200 and curr['close'] < fib_lvl:
            entry_side = "SHORT"
            entry_price = fib_lvl
            sl_price = high_p

    if not entry_side: return None
    
    return {
        "side": entry_side,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "stdev_score": 0.0, # Gardé pour compatibilité header
        "ts": rates[-1]["time"] 
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

def check_and_clean_expired_orders(symbol: str):
    orders = mt5.orders_get(symbol=symbol, magic=MAGIC_NUMBER)
    if not orders: return
    last_tick = mt5.symbol_info_tick(symbol)
    if last_tick is None: return
    server_time = last_tick.time 
    max_duration_sec = TIMEFRAME_MT5_MIN * 60 * MAX_WAIT_CANDLES
    for order in orders:
        order_age_sec = server_time - order.time_setup
        order_age_min = int(order_age_sec / 60)
        if order_age_sec > max_duration_sec:
            print(f"⌛ [EXPIRATION] Ordre {symbol} trop vieux ({order_age_min} min). Suppression...")
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "magic": MAGIC_NUMBER
            }
            res = mt5.order_send(request)

def place_mt5_order(symbol: str, setup: Dict[str, Any]):
    check_and_clean_expired_orders(symbol)
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0: return
    orders = mt5.orders_get(symbol=symbol)
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
    if lots == 0: return

    comm_cost_currency = lots * ESTIMATED_COMM_PER_LOT
    risk_monetaire = account_info.balance * RISK_PERCENT
    ratio_frais = (comm_cost_currency / risk_monetaire) * 100 if risk_monetaire > 0 else 0

    print("\n" + "="*60)
    print(f"📢 SETUP DÉTECTÉ: {symbol} ({setup['side']})")
    print("-" * 60)
    print(f"    Entry : {entry}  |  SL : {sl}  |  TP : {tp}")
    print(f"    Risk  : {risk_monetaire:.2f} USD (approx)")
    print(f"    Lots  : {lots}")
    print("-" * 60)
    print(f"    📊 AUDIT FRAIS (COMMISSION ONLY) :")
    print(f"    Coût Comm.      : {comm_cost_currency:.2f} USD")
    print(f"    ---------------------------")
    
    if ratio_frais > 50:
        print(f"    IMPACT / RISK : {ratio_frais:.2f}%  🔴 BLOQUÉ (Frais > 50%)")
        print("="*60 + "\n")
        return
    else:
        print(f"    IMPACT / RISK : {ratio_frais:.2f}%  🟢 OK")
    print("="*60 + "\n")

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "magic": MAGIC_NUMBER,
        "comment": ORDER_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,                 
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
        sys.exit(1)

    pairs = []
    filename = "session_pairs_e8markets.txt" if os.path.exists("session_pairs_e8markets.txt") else "pairs.txt"
    try:
        with open(filename, "r", newline="", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        for line in raw_lines:
            p = line.split(",")[-1].strip()
            if "pair" not in p.lower() and "type" not in p.lower():
                pairs.append(p)
    except Exception as e:
        sys.exit(1)

    print(f"🚀 Bot LIVE {TIMEFRAME_STR} lancé | Risk: {RISK_PERCENT*100}% | Pairs: {len(pairs)}")
    print(f"Stratégie: EMA 200 + FIB 61.8 | TF: {TIMEFRAME_STR}")
    print(f"DIRECTIONS : LONG={'✅' if ALLOW_LONGS else '❌'} | SHORT={'✅' if ALLOW_SHORTS else '❌'}")
    print(f"Audit Frais: ACTIF (Comm: {ESTIMATED_COMM_PER_LOT} / lot)")
    print(f"⚠️ Expiration Gérée par Script.")

    last_printed_ts = 0
    processed_setups = {}

    try:
        while True:
            if pairs:
                ref_pair = pairs[0] 
                ref_rates = fetch_last_candles_from_db(engine, ref_pair, limit=5)
                if ref_rates:
                    last_db_ts = ref_rates[-1]['time']
                    if last_db_ts != last_printed_ts:
                        db_time_str = datetime.fromtimestamp(last_db_ts/1000, tz=timezone.utc).strftime('%H:%M')
                        print(f"\n--- 🕰️ DERNIÈRE DATA EN BASE ({ref_pair}) : {db_time_str} UTC --------------------------------------------------------------------")
                        last_printed_ts = last_db_ts
            
            for pair in pairs:
                check_and_clean_expired_orders(pair)
                rates = fetch_last_candles_from_db(engine, pair, limit=1000)
                if not rates: continue
                # Passage des paramètres de direction ici
                setup = detect_setup_on_last_candle(rates, ALLOW_LONGS, ALLOW_SHORTS)
                if setup:
                    if processed_setups.get(pair) != setup['ts']:
                        place_mt5_order(pair, setup)
                        processed_setups[pair] = setup['ts']
            time.sleep(10)

    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()