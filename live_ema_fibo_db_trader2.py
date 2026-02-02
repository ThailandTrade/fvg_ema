#!/usr/bin/env python3
# live_fvg_db_trader_V8_STRICT_FULL_FIX.py

import time
import sys
import os
import statistics
import MetaTrader5 as mt5
import re
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, List, Any, Tuple

# BDD
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------- CONFIGURATION ----------
TIMEFRAME_STR = "5m"            # Timeframe de trading (LTF)
TREND_FILTER_TF = "5m"          # Strict Backtest
TIMEFRAME_MT5_MIN = 5
MAGIC_NUMBER = 888888
ORDER_COMMENT = "GoldenTrend"

# Heure de fermeture (UTC) - SEULEMENT LE VENDREDI
CLOSE_HOUR_UTC = 21
CLOSE_MINUTE_START = 50 

# Estimation des commissions
ESTIMATED_COMM_PER_LOT = 5.0 

# Paramètres Stratégie Golden Trend
MAX_WAIT_CANDLES = 72           # Strict Backtest
EMA_TREND_PERIOD = 200
FIB_RETREACEMENT = 0.62
SWING_CONFIRMATION_LAG = 5 
STDEV_PERIOD = 200              

DEFAULT_RR = 3.0                # Strict Backtest
RISK_PERCENT = 0.0005           # Strict Backtest
STDEV_THRESHOLD = 0.5
STDEV_MAX = 1.0

# --- FILTRES DE DIRECTION ---
ALLOW_LONGS = True    
ALLOW_SHORTS = False

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

# ---------- PARSING FICHIER ----------

def parse_pairs_with_rr(filename: str) -> List[Tuple[str, float]]:
    pairs_data = []
    if not os.path.exists(filename): return []

    try:
        with open(filename, "r", newline="", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        for line in lines:
            if line.lower().startswith("pair") or line.lower().startswith("type"): continue
            
            parts = line.split(',')
            p_name = parts[0].strip()
            rr_val = float(DEFAULT_RR)
            
            if len(parts) >= 2:
                try:
                    rr_val = float(parts[1].strip())
                    p_name = parts[0].strip()
                except ValueError:
                    p_name = parts[1].strip() # Format Type,Pair
            
            p_name = p_name.replace(" ", "")
            if p_name:
                pairs_data.append((p_name, rr_val))
    except: pass
    return pairs_data

# ---------- RÉCUPÉRATION DONNÉES ----------

def fetch_last_candles_from_db(engine, pair: str, timeframe: str, limit: int = 250) -> Optional[List[Dict[str, Any]]]:
    s_pair = sanitize_name(pair)
    s_tf = sanitize_name(timeframe)
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

# ---------- LOGIQUE DE DÉTECTION & FILTRE HTF ----------

def identify_swings(rates: List[Dict[str, Any]], lag: int) -> tuple[List[bool], List[bool]]:
    is_sh = [False] * len(rates)
    is_sl = [False] * len(rates)
    
    for i in range(lag, len(rates) - lag):
        current_high = rates[i]['high']
        current_low = rates[i]['low']
        
        local_high = -1.0
        local_low = 99999999.0
        
        for k in range(i - lag, i + lag + 1):
            if rates[k]['high'] > local_high: local_high = rates[k]['high']
            if rates[k]['low'] < local_low: local_low = rates[k]['low']
            
        if current_high == local_high:
            is_sh[i] = True
        if current_low == local_low:
            is_sl[i] = True
            
    return is_sh, is_sl

def calculate_htf_trend_structure(rates_htf: List[Dict[str, Any]]) -> int:
    if not rates_htf or len(rates_htf) < 200:
        return 0

    is_sh, is_sl = identify_swings(rates_htf, SWING_CONFIRMATION_LAG)
    
    last_h = -1.0; prev_h = -1.0
    last_l = -1.0; prev_l = -1.0
    current_trend = 0
    
    for i in range(len(rates_htf)):
        confirmed_idx = i - SWING_CONFIRMATION_LAG
        
        if confirmed_idx >= 0:
            if is_sh[confirmed_idx]:
                prev_h = last_h
                last_h = rates_htf[confirmed_idx]['high']
                
            if is_sl[confirmed_idx]:
                prev_l = last_l
                last_l = rates_htf[confirmed_idx]['low']
            
            if last_h > 0 and prev_h > 0 and last_l > 0 and prev_l > 0:
                if last_h > prev_h and last_l > prev_l:
                    current_trend = 1
                elif last_l < prev_l and last_h < prev_h:
                    current_trend = -1
                    
    return current_trend


def detect_setup_on_last_candle(rates: List[Dict[str, Any]], htf_trend: int, allow_longs: bool, allow_shorts: bool) -> Optional[Dict[str, Any]]:
    if len(rates) < EMA_TREND_PERIOD + 20: return None
    
    closes = [r['close'] for r in rates]
    ema_200 = calculate_ema_pandas_style(closes, EMA_TREND_PERIOD)
    
    curr = rates[-1]
    is_uptrend_local = curr['close'] > ema_200
    is_downtrend_local = curr['close'] < ema_200
    
    is_sh, is_sl = identify_swings(rates, SWING_CONFIRMATION_LAG)
    vision_limit = len(rates) - 1 - SWING_CONFIRMATION_LAG
    
    entry_side = ""
    entry_price = 0.0
    sl_price = 0.0

    # --- SETUP LONG ---
    if allow_longs and is_uptrend_local and htf_trend == 1:
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

    # --- SETUP SHORT ---
    elif allow_shorts and is_downtrend_local and htf_trend == -1:
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
        "stdev_score": 0.0,
        "ts": rates[-1]["time"] 
    }

# ---------- EXÉCUTION MT5 & AUDIT ----------

def qround(x: float, scale: int) -> float:
    return float(Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP))

def get_lot_size(symbol: str, entry: float, sl: float, risk_percent: float, balance: float):
    if entry == sl: return 0.0
    
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[ERR] Symbol info introuvable pour {symbol}")
        return 0.0

    if entry > sl:
        order_type = mt5.ORDER_TYPE_BUY
    else:
        order_type = mt5.ORDER_TYPE_SELL

    try:
        profit_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
    except Exception as e:
        print(f"[ERR] Erreur calcul profit MT5: {e}")
        return 0.0

    if profit_one_lot is None:
        print(f"[WARN] Fallback calcul manuel pour {symbol}")
        loss_per_lot_absolute = abs(entry - sl) * symbol_info.trade_contract_size
    else:
        loss_per_lot_absolute = abs(profit_one_lot)

    if loss_per_lot_absolute == 0:
        return 0.0

    risk_amount = balance * risk_percent
    raw_lots = risk_amount / loss_per_lot_absolute

    step = symbol_info.volume_step
    min_vol = symbol_info.volume_min
    max_vol = symbol_info.volume_max

    lots = round(raw_lots / step) * step
    
    if lots < min_vol: 
        return float(min_vol) 
        
    if lots > max_vol: 
        lots = max_vol

    return float(lots)

def manage_existing_orders(symbol: str, rates: List[Dict[str, Any]]):
    """
    Gestion Stricte : 
    1. Si Position Active -> On bloque (return True).
    2. Si Ordre Pending -> On gère (Timeout / Invalidation) et On bloque (return True).
    Retourne True si le symbole est "occupé" (Position ou Ordre en cours).
    """
    
    # 1. CHECK POSITIONS (Trade en cours)
    positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
    if positions:
        # Trade actif -> On ne fait rien, on ne cherche pas de setup.
        return True

    # 2. CHECK PENDING ORDERS
    orders = mt5.orders_get(symbol=symbol, magic=MAGIC_NUMBER)
    if not orders: 
        return False # Rien du tout, voie libre

    # Ici, on a des ordres pending, donc on est "occupé".
    # On en profite pour vérifier leur validité (Invalidation EMA / Timeout)
    
    if not rates: return True # Pas de data, mais ordres présents -> on bloque
    
    last_tick = mt5.symbol_info_tick(symbol)
    server_time = last_tick.time if last_tick else time.time()
    max_duration_sec = TIMEFRAME_MT5_MIN * 60 * MAX_WAIT_CANDLES
    
    curr_close = rates[-1]['close']
    ema_200 = calculate_ema_pandas_style([r['close'] for r in rates], EMA_TREND_PERIOD)
    
    for order in orders:
        # A. Timeout check
        order_age_sec = server_time - order.time_setup
        order_age_min = int(order_age_sec / 60)
        
        if order_age_sec > max_duration_sec:
            print(f"⌛ [EXPIRATION] Ordre {symbol} trop vieux ({order_age_min} min). Suppression...")
            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket, "magic": MAGIC_NUMBER})
            continue

        # B. Invalidation EMA Check
        invalidated = False
        if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP):
             if curr_close < ema_200: invalidated = True
        elif order.type in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP):
             if curr_close > ema_200: invalidated = True
             
        if invalidated:
            print(f"💀 [INVALIDATION] Ordre {symbol} annulé (Close vs EMA). Suppression...")
            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket, "magic": MAGIC_NUMBER})
            continue
            
    # Quoi qu'il arrive, si on avait des ordres (même s'ils viennent d'être supprimés dans ce tick),
    # on considère ce cycle comme "occupé" pour éviter de renvoyer un ordre instantanément sur la même bougie.
    return True

def place_mt5_order(symbol: str, setup: Dict[str, Any], rr_ratio: float):
    # 0. VÉRIFICATION PRIX LIVE
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return

    # --- AJOUT CRITIQUE : FILTRE SPREAD ---
    spread = tick.ask - tick.bid
    dist_entry_sl = abs(setup['entry_price'] - setup['sl_price'])
    # Si le spread représente plus de 25% de la distance du SL, c'est trop risqué (Scalping/Illiquide)
    # Cela évite de se faire exécuter instantanément au mauvais prix
    if dist_entry_sl > 0 and spread > (dist_entry_sl * 0.25):
        print(f"⚠️ [SPREAD WARNING] {symbol}: Spread ({spread:.5f}) > 25% du SL ({dist_entry_sl:.5f}). ABORT.")
        return
    # --------------------------------------

    entry = setup['entry_price']
    sl = setup['sl_price']
    order_type = None

    if setup['side'] == "LONG":
        if tick.ask > entry: order_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_BUY_STOP
            print(f"⚡ {symbol} LONG : Prix ({tick.ask}) < Entry ({entry}) -> Passage en BUY STOP")

    elif setup['side'] == "SHORT":
        if tick.bid < entry: order_type = mt5.ORDER_TYPE_SELL_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_SELL_STOP
            print(f"⚡ {symbol} SHORT : Prix ({tick.bid}) > Entry ({entry}) -> Passage en SELL STOP")

    # 1. Calculs
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info: return
    account_info = mt5.account_info()
    if not account_info: return

    dist_sl = abs(entry - sl)
    # MODIFICATION : UTILISATION DU RR DYNAMIQUE
    tp = entry + (dist_sl * rr_ratio) if setup['side'] == "LONG" else entry - (dist_sl * rr_ratio)

    scale = symbol_info.digits
    entry = qround(entry, scale)
    sl = qround(sl, scale)
    tp = qround(tp, scale)
    
    lots = get_lot_size(symbol, entry, sl, RISK_PERCENT, account_info.balance)
    if lots == 0: return

    # 3. Envoi (MODE GTC SANS EXPIRATION)
    print("\n" + "="*60)
    print(f"📢 SETUP DÉTECTÉ: {symbol} ({setup['side']}) [RR: {rr_ratio}]")
    print(f"    Mode  : {'LIMIT' if 'LIMIT' in str(order_type) else 'STOP (Catch-up)'}")
    print(f"    Entry : {entry}  |  SL : {sl}  |  TP : {tp}")
    print(f"    Risk  : {account_info.balance * RISK_PERCENT:.2f} USD | Lots : {lots}")
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
        "type_time": mt5.ORDER_TIME_GTC, # GTC CLASSIQUE
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        t_str = datetime.fromtimestamp(setup['ts']/1000, tz=timezone.utc).strftime('%H:%M')
        print(f"✅ [EXEC] {symbol} {setup['side']} (Bougie {t_str} UTC) | Entry: {entry} | SL: {sl} | Lots: {lots}")
    else:
        print(f"[MT5 ERROR] {symbol}: {res.comment} ({res.retcode})")


# ---------- FONCTION DE FERMETURE JOURNALIÈRE (MODIFIÉE VENDREDI SEULEMENT) ----------

def daily_market_close_guard():
    now_utc = datetime.now(timezone.utc)
    # 4 correspond à VENDREDI (0=Lundi, 6=Dimanche)
    if now_utc.weekday() == 4 and now_utc.hour == CLOSE_HOUR_UTC and now_utc.minute >= CLOSE_MINUTE_START:
        print(f"\n🛑 [WEEKEND CLOSE] Il est {now_utc.strftime('%H:%M')} UTC un VENDREDI. Fermeture forcée !")
        
        orders = mt5.orders_get(magic=MAGIC_NUMBER)
        if orders:
            for o in orders:
                print(f"    🗑️ Suppression ordre {o.symbol}...")
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
        
        positions = mt5.positions_get(magic=MAGIC_NUMBER)
        if positions:
            for p in positions:
                print(f"    👋 Fermeture position {p.symbol}...")
                tick = mt5.symbol_info_tick(p.symbol)
                if not tick: continue
                
                type_close = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price_close = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": type_close,
                    "position": p.ticket,
                    "price": price_close,
                    "magic": MAGIC_NUMBER,
                    "comment": "Weekend Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(request)
        
        print("💤 Pause du bot jusqu'à la réouverture... (Sleep 60s)")
        time.sleep(60)
        return True
    return False


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

    filename = "session_pairs_e8markets.txt" if os.path.exists("session_pairs_e8markets.txt") else "pairs.txt"
    pairs_data = parse_pairs_with_rr(filename)
    
    pairs = [p[0] for p in pairs_data]

    print(f"🚀 Bot LIVE {TIMEFRAME_STR} + HTF {TREND_FILTER_TF} lancé")
    print(f"Risk: {RISK_PERCENT*100}% | Pairs: {len(pairs)}")
    print(f"Fermeture forcée : VENDREDI à {CLOSE_HOUR_UTC}:{CLOSE_MINUTE_START} UTC")

    last_printed_ts = 0
    # processed_setups = {} # REMPLACÉ PAR LA LOGIQUE PLUS ROBUSTE CI-DESSOUS
    
    # DICTIONNAIRE MEMOIRE ANTI-MITRAILLAGE
    # Cle: Pair, Valeur: Timestamp de la bougie où on a déjà tenté un trade
    last_trade_candle_ts = {} 

    try:
        while True:
            if daily_market_close_guard():
                continue 

            if pairs:
                ref_pair = pairs[0] 
                ref_rates = fetch_last_candles_from_db(engine, ref_pair, TIMEFRAME_STR, limit=5)
                if ref_rates:
                    last_db_ts = ref_rates[-1]['time']
                    if last_db_ts != last_printed_ts:
                        db_time_str = datetime.fromtimestamp(last_db_ts/1000, tz=timezone.utc).strftime('%H:%M')
                        print(f"\n--- 🕰️ DERNIÈRE DATA EN BASE ({ref_pair} {TIMEFRAME_STR}) : {db_time_str} UTC -----------------------------------------------------------------")
                        last_printed_ts = last_db_ts
            
            for pair, specific_rr in pairs_data:
                
                # Fetch data nécessaire pour management ET detection
                # CORRECTIF 1: Limit augmentée à 5000 pour convergence EMA
                rates_ltf = fetch_last_candles_from_db(engine, pair, TIMEFRAME_STR, limit=5000)
                if not rates_ltf: continue

                # CORRECTIF 2 : ANTI-MITRAILLAGE
                # Si le timestamp de la dernière bougie est le même que celui où on a déjà tradé -> ON PASSE
                current_ts = rates_ltf[-1]['time']
                if last_trade_candle_ts.get(pair) == current_ts:
                    continue 

                # GESTION STRICTE : Position ou Pending -> On passe
                if manage_existing_orders(pair, rates_ltf):
                    continue
                
                # CORRECTIF 1 (Suite): Limit augmentée à 3000 pour convergence Trend
                rates_htf = fetch_last_candles_from_db(engine, pair, TREND_FILTER_TF, limit=3000)
                htf_trend = calculate_htf_trend_structure(rates_htf)
                
                if htf_trend == 0:
                    continue
                
                setup = detect_setup_on_last_candle(rates_ltf, htf_trend, ALLOW_LONGS, ALLOW_SHORTS)
                
                if setup:
                    # Plus besoin de verifier processed_setups car last_trade_candle_ts bloque tout en amont
                    rr_to_use = specific_rr if specific_rr is not None else DEFAULT_RR
                    place_mt5_order(pair, setup, rr_to_use)
                    
                    # CORRECTIF 2 : On marque la bougie comme "traitée"
                    last_trade_candle_ts[pair] = current_ts
                        
            time.sleep(10)

    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()