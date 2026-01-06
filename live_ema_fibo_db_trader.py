#!/usr/bin/env python3
# live_fvg_db_trader_30m_FIXED_DAILY_CLOSE_HTF.py

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
TIMEFRAME_STR = "5m"           # Timeframe de trading (LTF)
TREND_FILTER_TF = "1h"        # NOUVEAU : Timeframe du filtre de tendance (HTF)
TIMEFRAME_MT5_MIN = 5
MAGIC_NUMBER = 888888
ORDER_COMMENT = "GoldenTrend"

# Heure de fermeture (UTC)
CLOSE_HOUR_UTC = 21
CLOSE_MINUTE_START = 50 

# Estimation des commissions (Ex: 5 USD par lot Round-Turn)
ESTIMATED_COMM_PER_LOT = 5.0 

# Paramètres Stratégie Golden Trend
MAX_WAIT_CANDLES = 32         
EMA_TREND_PERIOD = 200
FIB_RETREACEMENT = 0.62
SWING_CONFIRMATION_LAG = 5 
STDEV_PERIOD = 200            

DEFAULT_RR = 2.0              
RISK_PERCENT = 0.001         
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

# ---------- RÉCUPÉRATION DONNÉES (MODIFIÉ POUR TF VARIABLE) ----------

def fetch_last_candles_from_db(engine, pair: str, timeframe: str, limit: int = 250) -> Optional[List[Dict[str, Any]]]:
    """
    Charge les bougies depuis la BDD.
    Accepte maintenant le paramètre 'timeframe' pour charger soit le 5m, soit le 30m.
    """
    s_pair = sanitize_name(pair)
    s_tf = sanitize_name(timeframe) # Utilise le TF passé en argument
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
        # print(f"[WARN] Fetch Error {pair} {timeframe}: {e}")
        return None

# ---------- LOGIQUE DE DÉTECTION & FILTRE HTF ----------

def identify_swings(rates: List[Dict[str, Any]], lag: int) -> tuple[List[bool], List[bool]]:
    is_sh = [False] * len(rates)
    is_sl = [False] * len(rates)
    # Optimisation: on peut limiter la boucle si on veut, mais sur 1000 bougies c'est rapide
    for i in range(lag, len(rates) - lag):
        # Vérifie si le point i est le plus haut/bas de sa fenêtre locale
        current_high = rates[i]['high']
        current_low = rates[i]['low']
        
        # Fenêtre : [i-lag ... i ... i+lag]
        local_high = -1.0
        local_low = 99999999.0
        
        # Boucle manuelle pour éviter création de sous-listes coûteuses
        for k in range(i - lag, i + lag + 1):
            if rates[k]['high'] > local_high: local_high = rates[k]['high']
            if rates[k]['low'] < local_low: local_low = rates[k]['low']
            
        if current_high == local_high:
            is_sh[i] = True
        if current_low == local_low:
            is_sl[i] = True
            
    return is_sh, is_sl

def calculate_htf_trend_structure(rates_htf: List[Dict[str, Any]]) -> int:
    """
    Analyse les données HTF (ex: 30m) pour déterminer la tendance structurelle.
    Retourne: 1 (Bullish), -1 (Bearish), 0 (Neutre).
    """
    if not rates_htf or len(rates_htf) < 200:
        return 0

    is_sh, is_sl = identify_swings(rates_htf, SWING_CONFIRMATION_LAG)
    
    last_h = -1.0; prev_h = -1.0
    last_l = -1.0; prev_l = -1.0
    current_trend = 0
    
    # On parcourt toute la liste HTF pour mettre à jour l'état de la tendance bougie par bougie
    # Cela assure qu'on a l'état actuel à la dernière bougie fermée
    for i in range(len(rates_htf)):
        # Un swing est confirmé à i - LAG
        confirmed_idx = i - SWING_CONFIRMATION_LAG
        
        if confirmed_idx >= 0:
            if is_sh[confirmed_idx]:
                prev_h = last_h
                last_h = rates_htf[confirmed_idx]['high']
                
            if is_sl[confirmed_idx]:
                prev_l = last_l
                last_l = rates_htf[confirmed_idx]['low']
            
            # Mise à jour tendance
            if last_h > 0 and prev_h > 0 and last_l > 0 and prev_l > 0:
                if last_h > prev_h and last_l > prev_l:
                    current_trend = 1
                elif last_l < prev_l and last_h < prev_h:
                    current_trend = -1
                    
    return current_trend


def detect_setup_on_last_candle(rates: List[Dict[str, Any]], htf_trend: int, allow_longs: bool, allow_shorts: bool) -> Optional[Dict[str, Any]]:
    """
    Détecte le setup sur le LTF (5m) en prenant en compte le filtre HTF (30m).
    """
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
    # Condition ajoutée : and htf_trend == 1
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
    # Condition ajoutée : and htf_trend == -1
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

# ---------- EXÉCUTION MT5 & AUDIT (Inchangé) ----------

def qround(x: float, scale: int) -> float:
    return float(Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP))

def get_lot_size(symbol: str, entry: float, sl: float, risk_percent: float, balance: float):
    """
    Calcul des lots robuste utilisant mt5.order_calc_profit pour gérer
    correctement les tailles de contrats (XAU, XAG, Indices, Forex).
    """
    if entry == sl: return 0.0
    
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[ERR] Symbol info introuvable pour {symbol}")
        return 0.0

    # 1. Déterminer le sens pour la simulation (Long ou Short)
    # Si Entry > SL, c'est un achat (on perd si ça descend).
    # Si Entry < SL, c'est une vente (on perd si ça monte).
    if entry > sl:
        order_type = mt5.ORDER_TYPE_BUY
    else:
        order_type = mt5.ORDER_TYPE_SELL

    # 2. Demander à MT5 la perte en devise du compte pour 1.0 LOT standard
    # Cette fonction gère automatiquement ContractSize (100 pour l'or), TickValue, etc.
    try:
        profit_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
    except Exception as e:
        print(f"[ERR] Erreur calcul profit MT5: {e}")
        return 0.0

    # Si la fonction échoue ou renvoie None (ex: marché fermé ou data manquante)
    if profit_one_lot is None:
        # TENTATIVE DE FALLBACK (Calcul manuel 'Contract Size')
        # Pour XAUUSD/USD, Loss = Distance * ContractSize
        print(f"[WARN] Fallback calcul manuel pour {symbol}")
        loss_per_lot_absolute = abs(entry - sl) * symbol_info.trade_contract_size
    else:
        # profit_one_lot est négatif car c'est une perte, on prend l'absolu
        loss_per_lot_absolute = abs(profit_one_lot)

    if loss_per_lot_absolute == 0:
        return 0.0

    # 3. Calcul du volume
    risk_amount = balance * risk_percent
    raw_lots = risk_amount / loss_per_lot_absolute

    # 4. Normalisation (Step, Min, Max)
    step = symbol_info.volume_step
    min_vol = symbol_info.volume_min
    max_vol = symbol_info.volume_max

    # Arrondi au step près (ex: 0.01)
    lots = round(raw_lots / step) * step
    
    # Sécurité finale
    if lots < min_vol: 
        # Si le risque calculé est inférieur au lot min, on retourne 0 (ou min_vol si tu es agressif)
        # Ici on retourne min_vol pour ne pas bloquer les trades sur petits comptes, 
        # mais attention le risque sera > risk_percent.
        return float(min_vol) 
        
    if lots > max_vol: 
        lots = max_vol

    # Debug optionnel pour vérifier dans la console
    # print(f"DEBUG {symbol}: Risk=${risk_amount:.2f} | Loss/1Lot=${loss_per_lot_absolute:.2f} -> Lots Calc: {lots}")

    return float(lots)

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
            mt5.order_send(request)

def place_mt5_order(symbol: str, setup: Dict[str, Any]):
    # 0. VÉRIFICATION PRIX LIVE
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return

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
    tp = entry + (dist_sl * DEFAULT_RR) if setup['side'] == "LONG" else entry - (dist_sl * DEFAULT_RR)

    scale = symbol_info.digits
    entry = qround(entry, scale)
    sl = qround(sl, scale)
    tp = qround(tp, scale)
    
    lots = get_lot_size(symbol, entry, sl, RISK_PERCENT, account_info.balance)
    if lots == 0: return

    # 2. Nettoyage
    check_and_clean_expired_orders(symbol)
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0: return
    orders = mt5.orders_get(symbol=symbol)
    if orders:
        for order in orders:
            mt5.order_send({ "action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket, "magic": MAGIC_NUMBER })
            print(f"♻️ [UPDATE] {symbol}: Ancien ordre supprimé pour mise à jour.")

    # 3. Envoi
    print("\n" + "="*60)
    print(f"📢 SETUP DÉTECTÉ: {symbol} ({setup['side']})")
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
        "type_time": mt5.ORDER_TIME_GTC,                  
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        t_str = datetime.fromtimestamp(setup['ts']/1000, tz=timezone.utc).strftime('%H:%M')
        print(f"✅ [EXEC] {symbol} {setup['side']} (Bougie {t_str} UTC) | Entry: {entry} | SL: {sl} | Lots: {lots}")
    else:
        print(f"[MT5 ERROR] {symbol}: {res.comment} ({res.retcode})")


# ---------- FONCTION DE FERMETURE JOURNALIÈRE (Inchangée) ----------

def daily_market_close_guard():
    now_utc = datetime.now(timezone.utc)
    if now_utc.hour == CLOSE_HOUR_UTC and now_utc.minute >= CLOSE_MINUTE_START:
        print(f"\n🛑 [DAILY CLOSE] Il est {now_utc.strftime('%H:%M')} UTC. Fermeture journalière forcée !")
        
        orders = mt5.orders_get(magic=MAGIC_NUMBER)
        if orders:
            for o in orders:
                print(f"   🗑️ Suppression ordre {o.symbol}...")
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
        
        positions = mt5.positions_get(magic=MAGIC_NUMBER)
        if positions:
            for p in positions:
                print(f"   👋 Fermeture position {p.symbol}...")
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
                    "comment": "Daily Close",
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

    print(f"🚀 Bot LIVE {TIMEFRAME_STR} + HTF {TREND_FILTER_TF} lancé")
    print(f"Risk: {RISK_PERCENT*100}% | Pairs: {len(pairs)}")
    print(f"Fermeture journalière forcée à : {CLOSE_HOUR_UTC}:{CLOSE_MINUTE_START} UTC")

    last_printed_ts = 0
    processed_setups = {}

    try:
        while True:
            # 1. SÉCURITÉ FERMETURE
            if daily_market_close_guard():
                continue 

            # 2. LOGIQUE NORMALE
            if pairs:
                ref_pair = pairs[0] 
                # On vérifie la fraîcheur des données LTF (5m) pour l'affichage
                ref_rates = fetch_last_candles_from_db(engine, ref_pair, TIMEFRAME_STR, limit=5)
                if ref_rates:
                    last_db_ts = ref_rates[-1]['time']
                    if last_db_ts != last_printed_ts:
                        db_time_str = datetime.fromtimestamp(last_db_ts/1000, tz=timezone.utc).strftime('%H:%M')
                        print(f"\n--- 🕰️ DERNIÈRE DATA EN BASE ({ref_pair} {TIMEFRAME_STR}) : {db_time_str} UTC -----------------------------------------------------------------")
                        last_printed_ts = last_db_ts
            
            for pair in pairs:
                check_and_clean_expired_orders(pair)
                
                # A. Récupération HTF (30m) & Calcul Tendance
                rates_htf = fetch_last_candles_from_db(engine, pair, TREND_FILTER_TF, limit=500)
                htf_trend = calculate_htf_trend_structure(rates_htf)
                
                # B. Si tendance neutre, pas besoin de charger le LTF (économie ressources)
                if htf_trend == 0:
                    continue
                
                # C. Récupération LTF (5m) & Détection
                rates_ltf = fetch_last_candles_from_db(engine, pair, TIMEFRAME_STR, limit=1000)
                if not rates_ltf: continue
                
                setup = detect_setup_on_last_candle(rates_ltf, htf_trend, ALLOW_LONGS, ALLOW_SHORTS)
                
                if setup:
                    if processed_setups.get(pair) != setup['ts']:
                        place_mt5_order(pair, setup)
                        processed_setups[pair] = setup['ts']
                        
            time.sleep(10)

    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()