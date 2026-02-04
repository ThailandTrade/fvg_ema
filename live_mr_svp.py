#!/usr/bin/env python3
"""
Live Trading - VP Failed Breakout Strategy
ALIGNÉ EXACTEMENT SUR LE BACKTEST
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import psycopg2
import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from sqlalchemy import create_engine

load_dotenv()

# =============================================================================
# CONFIGURATION - IDENTIQUE AU BACKTEST
# =============================================================================

RISK_PERCENT = 0.001  # 0.1% par trade

# Mode de target
TP_MODE = "POC"  # "FIXED_RR" = R:R fixe | "POC" = Target au POC
TARGET_RR = 3.0
MIN_RR = 2.0     # R:R minimum pour ENTRER en position

# Trailing
USE_TRAILING = True
TP1_RR = 1.2      # R:R ou on ferme 50% et met BE

# Filtres globaux
FILTER_ENTRY_VS_POC = True
USE_BREAKOUT_DURATION_FILTER = True
MAX_BREAKOUT_DURATION_MINUTES = 3
USE_VP_STRUCTURE_FILTER = True
MIN_POC_STRENGTH = 2.5
USE_VP_SHAPE_FILTER = True
EXCLUDED_VP_SHAPES = ["P-SHAPE"]

# Reset VP
RESET_VP_PER_SESSION = True

# =============================================================================
# CONFIGURATION DES HEURES DE SESSION (UTC)
# =============================================================================
SESSIONS_CONFIG = {
    'TOKYO':  {'start': 0,    'end': 4},     # 0h00 - 4h00 UTC
    'LONDON': {'start': 8,    'end': 13},    # 8h00 - 13h00 UTC
    'NY':     {'start': 14.5, 'end': 21},    # 14h30 - 21h00 UTC
}

# =============================================================================
# CONFIGURATION PAR ASSET - sl_offset = 0.10 pour tous (comme backtest)
# =============================================================================
ASSETS = [
    {
        'enabled': True,
        'symbol': 'XAUUSD',
        'mt5_symbol': 'XAUUSD',
        'candle_table': 'candles_mt5_xauusd_1m',
        'tick_table': 'market_ticks_xauusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
    },
    {
        'enabled': False,
        'symbol': 'JP225.cash',
        'mt5_symbol': 'JP225.cash',
        'candle_table': 'candles_mt5_jp225_cash_1m',
        'tick_table': 'market_ticks_jp225',
        'tick_size': 1.0,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': True},
    },
    {
        'enabled': False,
        'symbol': 'UK100.cash',
        'mt5_symbol': 'UK100.cash',
        'candle_table': 'candles_mt5_uk100_cash_1m',
        'tick_table': 'market_ticks_uk100',
        'tick_size': 1.0,
        'va_percent': 0.70,
        'allow_long': False,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False},
    },
    {
        'enabled': False,
        'symbol': 'GER40.cash',
        'mt5_symbol': 'GER40.cash',
        'candle_table': 'candles_mt5_ger40_cash_1m',
        'tick_table': 'market_ticks_ger40',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': True, 'LONDON': False, 'NY': True},
    },
    {
        'enabled': False,
        'symbol': 'US30.cash',
        'mt5_symbol': 'US30.cash',
        'candle_table': 'candles_mt5_us30_cash_1m',
        'tick_table': 'market_ticks_us30',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': False,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False},
    },
    {
        'enabled': False,
        'symbol': 'XAUAUD',
        'mt5_symbol': 'XAUAUD',
        'candle_table': 'candles_mt5_xauaud_1m',
        'tick_table': 'market_ticks_xauaud',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': False,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False},
    },
    {
        'enabled': False,
        'symbol': 'XPDUSD',
        'mt5_symbol': 'XPDUSD',
        'candle_table': 'candles_mt5_xpdusd_1m',
        'tick_table': 'market_ticks_xpdusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
    },
    {
        'enabled': False,
        'symbol': 'XPTUSD',
        'mt5_symbol': 'XPTUSD',
        'candle_table': 'candles_mt5_xptusd_1m',
        'tick_table': 'market_ticks_xptusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sl_offset': 0.10,
        'sessions': {'TOKYO': True, 'LONDON': False, 'NY': True},
    },
]

# =============================================================================
# PARAMETRES LIVE
# =============================================================================
LOOP_INTERVAL_SECONDS = 5
STATE_FILE = "vp_failed_breakout_state.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE
# =============================================================================

def get_pg_engine():
    db_url = f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    return create_engine(db_url)


def fetch_candles_from_db(engine, table_name: str, limit: int = 60):
    query = f"SELECT ts, open, high, low, close FROM {table_name} ORDER BY ts DESC LIMIT {limit}"
    df = pd.read_sql(query, engine)
    if df.empty:
        return []
    df = df.sort_values('ts').reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.to_dict('records')


def fetch_candles_since(engine, table_name: str, since_dt: datetime):
    """Charge toutes les candles depuis une datetime"""
    ts_start = int(since_dt.timestamp() * 1000)
    query = f"SELECT ts, open, high, low, close FROM {table_name} WHERE ts >= {ts_start} ORDER BY ts ASC"
    df = pd.read_sql(query, engine)
    if df.empty:
        return []
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.to_dict('records')


def fetch_ticks_from_db(engine, tick_table: str, start_time: datetime, end_time: datetime):
    """
    Récupère les ticks de start_time (inclus) à end_time (EXCLUS)
    Aligné sur le backtest qui utilise floor('T') pour grouper par minute
    """
    query = f"""
        SELECT time, last as price, volume 
        FROM {tick_table} 
        WHERE time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}' 
        AND time < '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY time ASC
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return np.array([]), np.array([])
    return df['price'].values, df['volume'].values


# =============================================================================
# VOLUME PROFILE
# =============================================================================

class IncrementalVolumeProfile:
    def __init__(self, tick_size: float = 0.01, va_percent: float = 0.70):
        self.tick_size = tick_size
        self.va_percent = va_percent
        self.reset()
    
    def reset(self):
        self.profile = defaultdict(float)
        self.total_volume = 0.0
        self._cache_valid = False
        self._cached_poc = None
        self._cached_vah = None
        self._cached_val = None
        self._cached_poc_strength = None
    
    def add_ticks(self, prices: np.ndarray, volumes: np.ndarray):
        if len(prices) == 0:
            return
        price_bins = np.round(prices / self.tick_size) * self.tick_size
        unique_bins, indices = np.unique(price_bins, return_inverse=True)
        bin_volumes = np.bincount(indices, weights=volumes)
        for price_bin, vol in zip(unique_bins, bin_volumes):
            self.profile[price_bin] += vol
            self.total_volume += vol
        self._cache_valid = False
    
    def _compute_value_area(self):
        if not self.profile:
            self._cached_poc = self._cached_vah = self._cached_val = None
            self._cached_poc_strength = None
            self._cache_valid = True
            return
        sorted_bins = sorted(self.profile.keys())
        volumes = np.array([self.profile[b] for b in sorted_bins])
        poc_idx = np.argmax(volumes)
        self._cached_poc = sorted_bins[poc_idx]
        poc_volume = volumes[poc_idx]
        avg_volume = np.mean(volumes)
        self._cached_poc_strength = poc_volume / avg_volume if avg_volume > 0 else 0
        target_volume = self.total_volume * self.va_percent
        current_volume = volumes[poc_idx]
        up_idx, down_idx = poc_idx + 1, poc_idx - 1
        while current_volume < target_volume:
            vol_up = volumes[up_idx] if up_idx < len(volumes) else 0
            vol_down = volumes[down_idx] if down_idx >= 0 else 0
            if vol_up == 0 and vol_down == 0:
                break
            if vol_up >= vol_down:
                current_volume += vol_up
                up_idx += 1
            else:
                current_volume += vol_down
                down_idx -= 1
        self._cached_vah = sorted_bins[min(up_idx - 1, len(sorted_bins) - 1)]
        self._cached_val = sorted_bins[max(down_idx + 1, 0)]
        self._cache_valid = True
    
    def get_levels(self):
        if not self._cache_valid:
            self._compute_value_area()
        return self._cached_poc, self._cached_vah, self._cached_val
    
    def get_poc_strength(self):
        if not self._cache_valid:
            self._compute_value_area()
        return self._cached_poc_strength
    
    def get_profile_shape(self):
        if not self._cache_valid:
            self._compute_value_area()
        if self._cached_poc_strength is None or self._cached_poc_strength < 1.3:
            return "FLAT"
        if self._cached_poc is None or self._cached_vah is None or self._cached_val is None:
            return "FLAT"
        va_range = self._cached_vah - self._cached_val
        if va_range == 0:
            return "FLAT"
        poc_position = (self._cached_poc - self._cached_val) / va_range
        if poc_position > 0.66:
            return "P-SHAPE"
        elif poc_position < 0.33:
            return "B-SHAPE"
        else:
            return "D-SHAPE"


# =============================================================================
# SESSION HELPERS
# =============================================================================

def get_session(dt):
    h = dt.hour + dt.minute / 60.0
    for sess_name, cfg in SESSIONS_CONFIG.items():
        if cfg['start'] <= h < cfg['end']:
            return sess_name
    return "AUTRE"


def is_session_start(dt):
    """Identique au backtest: current_time == cfg['start']"""
    current_time = dt.hour + dt.minute / 60.0
    for sess_name, cfg in SESSIONS_CONFIG.items():
        if current_time == cfg['start']:
            return sess_name
    return None


# =============================================================================
# MT5 HELPERS
# =============================================================================

def qround(price: float, tick_size: float) -> float:
    d = Decimal(str(price))
    t = Decimal(str(tick_size))
    return float(d.quantize(t, rounding=ROUND_DOWN))


def get_lot_size(symbol: str, entry: float, sl: float, risk_amount: float) -> float:
    if entry == sl:
        return 0.0
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"[{symbol}] Symbol info not found")
        return 0.0
    
    if entry > sl:
        order_type = mt5.ORDER_TYPE_BUY
    else:
        order_type = mt5.ORDER_TYPE_SELL
    
    try:
        profit_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
    except Exception as e:
        logger.error(f"[{symbol}] order_calc_profit error: {e}")
        return 0.0
    
    if profit_one_lot is None:
        logger.warning(f"[{symbol}] Fallback manual lot calculation")
        loss_per_lot = abs(entry - sl) * symbol_info.trade_contract_size
    else:
        loss_per_lot = abs(profit_one_lot)
    
    if loss_per_lot == 0:
        return 0.0
    
    raw_lots = risk_amount / loss_per_lot
    
    step = symbol_info.volume_step
    min_vol = symbol_info.volume_min
    max_vol = symbol_info.volume_max
    
    lots = round(raw_lots / step) * step
    
    if lots < min_vol:
        lots = min_vol
    if lots > max_vol:
        lots = max_vol
    
    return float(lots)


def has_position_or_order(symbol: str) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        return True
    orders = mt5.orders_get(symbol=symbol)
    if orders and len(orders) > 0:
        return True
    return False


def manage_tp1_to_be(symbol: str, tick_size: float):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    our_positions = [p for p in positions if "VP_FailedBO" in (p.comment or "")]
    tp2_positions = [p for p in our_positions if "_TP2" in (p.comment or "")]
    tp1_positions = [p for p in our_positions if "_TP1" in (p.comment or "")]
    
    if len(tp1_positions) == 0 and len(tp2_positions) == 1:
        pos = tp2_positions[0]
        entry_price = pos.price_open
        current_sl = pos.sl
        
        if abs(current_sl - entry_price) > tick_size * 2:
            new_sl = qround(entry_price, tick_size)
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[{symbol}] TP1 hit - SL moved to BE @ {new_sl}")
            else:
                logger.error(f"[{symbol}] Failed to move SL to BE: {result.retcode}")


def place_market_order(symbol: str, order_type: str, entry: float, sl: float, tp1: float, tp2: float,
                       risk_amount: float, tick_size: float, comment: str = "VP_FailedBO") -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"[{symbol}] Symbol info not found")
        return False
    
    sl = qround(sl, tick_size)
    tp1 = qround(tp1, tick_size)
    tp2 = qround(tp2, tick_size)
    
    risk_per_order = risk_amount / 2
    
    lot1 = get_lot_size(symbol, entry, sl, risk_per_order)
    lot2 = get_lot_size(symbol, entry, sl, risk_per_order)
    
    if lot1 <= 0 or lot2 <= 0:
        logger.error(f"[{symbol}] Invalid lot size: lot1={lot1}, lot2={lot2}")
        return False
    
    mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
    price = info.ask if order_type == "BUY" else info.bid
    
    logger.info(f"[{symbol}] ====== PLACING TRADE ======")
    logger.info(f"[{symbol}] Direction: {order_type} | Entry: {price} | SL: {sl}")
    logger.info(f"[{symbol}] TP1: {tp1} | TP2: {tp2}")
    logger.info(f"[{symbol}] Risk/order: ${risk_per_order:.2f} | Lot1: {lot1} | Lot2: {lot2}")
    
    success_count = 0
    
    request1 = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot1,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp1,
        "deviation": 20,
        "magic": 123456,
        "comment": f"{comment}_TP1",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result1 = mt5.order_send(request1)
    if result1.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{symbol}] TP1 FILLED: {order_type} {lot1} lots @ {result1.price}")
        success_count += 1
    else:
        logger.error(f"[{symbol}] TP1 FAILED: {result1.retcode} - {result1.comment}")
    
    request2 = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot2,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp2,
        "deviation": 20,
        "magic": 123456,
        "comment": f"{comment}_TP2",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result2 = mt5.order_send(request2)
    if result2.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{symbol}] TP2 FILLED: {order_type} {lot2} lots @ {result2.price}")
        success_count += 1
    else:
        logger.error(f"[{symbol}] TP2 FAILED: {result2.retcode} - {result2.comment}")
    
    logger.info(f"[{symbol}] ====== TRADE COMPLETE: {success_count}/2 orders filled ======")
    return success_count > 0


def daily_market_close_guard(candle_dt: datetime) -> bool:
    if candle_dt.weekday() == 4 and candle_dt.hour == 21 and candle_dt.minute >= 50:
        logger.warning("Friday market close - closing all")
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": "Friday close",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
        orders = mt5.orders_get()
        if orders:
            for order in orders:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
        return True
    return False


# =============================================================================
# ASSET STATE - Sans WAITING_REENTRY (comme backtest)
# =============================================================================

class AssetState:
    def __init__(self, config: dict):
        self.config = config
        self.symbol = config['symbol']
        self.mt5_symbol = config['mt5_symbol']
        self.vp = IncrementalVolumeProfile(tick_size=config['tick_size'], va_percent=config['va_percent'])
        self.state = "INSIDE"  # INSIDE, BREAKOUT_UP, BREAKOUT_DOWN (pas de WAITING_REENTRY)
        self.swing_extreme = 0.0
        self.breakout_time = None
        self.breakout_price = None
        self.current_session = None
        self.last_trade_candle_ts = 0
        self.last_candle_ts = 0
    
    def reset_vp(self):
        self.vp.reset()
        self.state = "INSIDE"
        self.swing_extreme = 0.0
        self.breakout_time = None
        self.breakout_price = None
    
    def to_dict(self) -> dict:
        return {
            'state': self.state,
            'swing_extreme': self.swing_extreme,
            'breakout_time': self.breakout_time.isoformat() if self.breakout_time else None,
            'breakout_price': self.breakout_price,
            'current_session': self.current_session,
            'last_trade_candle_ts': self.last_trade_candle_ts,
            'last_candle_ts': self.last_candle_ts,
        }
    
    def from_dict(self, data: dict):
        self.state = data.get('state', 'INSIDE')
        self.swing_extreme = data.get('swing_extreme', 0.0)
        bt = data.get('breakout_time')
        self.breakout_time = datetime.fromisoformat(bt) if bt else None
        self.breakout_price = data.get('breakout_price')
        self.current_session = data.get('current_session')
        self.last_trade_candle_ts = data.get('last_trade_candle_ts', 0)
        self.last_candle_ts = data.get('last_candle_ts', 0)


def save_all_states(asset_states: dict):
    data = {symbol: state.to_dict() for symbol, state in asset_states.items()}
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_all_states(asset_states: dict):
    if not os.path.exists(STATE_FILE):
        logger.info("No state file found, starting fresh")
        return
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        for symbol, saved in data.items():
            if symbol in asset_states:
                asset_states[symbol].from_dict(saved)
                logger.info(f"[{symbol}] State restored: {saved['state']} | swing: {saved['swing_extreme']}")
    except Exception as e:
        logger.error(f"Failed to load state: {e}")


# =============================================================================
# MAIN TRADING LOGIC - IDENTIQUE AU BACKTEST
# =============================================================================

def detect_and_trade(asset_state: AssetState, engine, account_balance: float) -> dict:
    config = asset_state.config
    symbol = config['symbol']
    mt5_symbol = config['mt5_symbol']
    
    result = {
        'new_candle': False,
        'symbol': symbol,
        'candle_dt': None,
        'close': None,
        'high': None,
        'low': None,
        'session': None,
        'state_before': asset_state.state,
        'state_after': None,
        'poc': None,
        'vah': None,
        'val': None,
        'poc_strength': None,
        'vp_shape': None,
        'swing_extreme': asset_state.swing_extreme,
        'event': None,
        'event_details': None,
        'has_position': False,
    }
    
    # Check position existante
    if has_position_or_order(mt5_symbol):
        result['has_position'] = True
        result['event'] = "POSITION_ACTIVE"
        return result
    
    # Charger les candles
    candles = fetch_candles_from_db(engine, config['candle_table'], limit=5)
    if not candles:
        result['event'] = "NO_DATA"
        return result
    
    last_candle = candles[-1]
    close = last_candle['close']
    high = last_candle['high']
    low = last_candle['low']
    candle_dt = last_candle['dt']
    candle_ts = last_candle['ts']
    
    result['candle_dt'] = candle_dt
    result['close'] = close
    result['high'] = high
    result['low'] = low
    
    # Vérifier si c'est une nouvelle bougie
    if asset_state.last_candle_ts == candle_ts:
        result['new_candle'] = False
        return result
    
    result['new_candle'] = True
    asset_state.last_candle_ts = candle_ts
    
    # Déterminer la session
    curr_sess = get_session(candle_dt)
    result['session'] = curr_sess
    asset_sessions = config.get('sessions', {})
    
    # Hors session -> INSIDE (comme backtest)
    if not asset_sessions.get(curr_sess, False):
        asset_state.state = "INSIDE"
        result['state_after'] = "INSIDE"
        result['event'] = "OUT_OF_SESSION"
        return result
    
    # Reset VP au debut de session (comme backtest)
    if RESET_VP_PER_SESSION:
        session_start = is_session_start(candle_dt)
        if session_start and asset_sessions.get(session_start, False):
            asset_state.reset_vp()
            asset_state.current_session = session_start
            result['event'] = "SESSION_START"
            result['event_details'] = session_start
    
    # Construire VP depuis debut de session jusqu'à FIN de la bougie actuelle
    # Aligné sur backtest: à la bougie N, on inclut les ticks de 00:00 à N:59.999
    session_cfg = SESSIONS_CONFIG.get(curr_sess, {})
    session_start_hour = session_cfg.get('start', 0)
    session_start_dt = candle_dt.replace(hour=int(session_start_hour), minute=int((session_start_hour % 1) * 60), second=0, microsecond=0)
    if session_start_dt > candle_dt:
        session_start_dt -= timedelta(days=1)
    
    # end_dt = début de la minute SUIVANTE (pour inclure toute la minute actuelle)
    end_dt = candle_dt + timedelta(minutes=1)
    
    prices, volumes = fetch_ticks_from_db(engine, config['tick_table'], session_start_dt, end_dt)
    
    if len(prices) > 0:
        asset_state.vp.reset()
        asset_state.vp.add_ticks(prices, volumes)
    
    poc, vah, val = asset_state.vp.get_levels()
    if poc is None:
        result['event'] = "NO_VP"
        result['event_details'] = f"0 ticks from {session_start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')}"
        result['state_after'] = asset_state.state
        return result
    
    poc_strength = asset_state.vp.get_poc_strength()
    vp_shape = asset_state.vp.get_profile_shape()
    
    result['poc'] = poc
    result['vah'] = vah
    result['val'] = val
    result['poc_strength'] = poc_strength
    result['vp_shape'] = vp_shape
    
    state = asset_state.state
    
    # ==========================================================================
    # STATE MACHINE - IDENTIQUE AU BACKTEST
    # ==========================================================================
    
    if state == "INSIDE":
        if close > vah:
            asset_state.state = "BREAKOUT_UP"
            asset_state.swing_extreme = high
            asset_state.breakout_time = candle_dt
            asset_state.breakout_price = close
            result['event'] = "BREAKOUT_UP"
            result['swing_extreme'] = high
        elif close < val:
            asset_state.state = "BREAKOUT_DOWN"
            asset_state.swing_extreme = low
            asset_state.breakout_time = candle_dt
            asset_state.breakout_price = close
            result['event'] = "BREAKOUT_DOWN"
            result['swing_extreme'] = low
        else:
            result['event'] = "INSIDE"
    
    elif state == "BREAKOUT_UP":
        # Mettre à jour swing_extreme (comme backtest: max(swing_extreme, high))
        asset_state.swing_extreme = max(asset_state.swing_extreme, high)
        result['swing_extreme'] = asset_state.swing_extreme
        
        if close < vah:
            # Failed breakout UP - évaluer SHORT (comme backtest)
            breakout_duration_min = (candle_dt - asset_state.breakout_time).total_seconds() / 60.0
            
            # Flags pour les filtres (comme backtest)
            duration_ok = True
            poc_strength_ok = True
            vp_shape_ok = True
            
            # Filtre duration
            if USE_BREAKOUT_DURATION_FILTER:
                if breakout_duration_min >= MAX_BREAKOUT_DURATION_MINUTES:
                    duration_ok = False
                    result['event'] = "FILTERED_DURATION"
                    result['event_details'] = f"{breakout_duration_min:.1f}min >= {MAX_BREAKOUT_DURATION_MINUTES}min"
            
            # Filtre POC strength
            if USE_VP_STRUCTURE_FILTER:
                if poc_strength is None or poc_strength < MIN_POC_STRENGTH:
                    poc_strength_ok = False
                    if duration_ok:  # Ne log que si pas déjà filtré
                        result['event'] = "FILTERED_POC_STRENGTH"
                        result['event_details'] = f"{poc_strength:.2f}x < {MIN_POC_STRENGTH}x"
            
            # Filtre VP shape
            if USE_VP_SHAPE_FILTER:
                if vp_shape in EXCLUDED_VP_SHAPES:
                    vp_shape_ok = False
                    if duration_ok and poc_strength_ok:
                        result['event'] = "FILTERED_VP_SHAPE"
                        result['event_details'] = vp_shape
            
            # Tous les filtres OK + allow_short (comme backtest)
            if duration_ok and poc_strength_ok and vp_shape_ok and config['allow_short']:
                sl = asset_state.swing_extreme + config['sl_offset']
                risk = sl - close
                
                if TP_MODE == "POC":
                    tp = poc
                    actual_rr = (close - tp) / risk if risk > 0 else 0
                    # Filtre RR aberrant (comme backtest: > 10)
                    if actual_rr > 10:
                        result['event'] = "FILTERED_RR_ABERRANT"
                        result['event_details'] = f"RR={actual_rr:.1f}"
                        asset_state.state = "INSIDE"
                        result['state_after'] = "INSIDE"
                        return result
                else:
                    tp = close - (risk * TARGET_RR)
                    actual_rr = TARGET_RR
                
                poc_ok = (close >= poc) if FILTER_ENTRY_VS_POC else True
                rr_ok = actual_rr >= MIN_RR
                
                if risk > 0 and tp >= val and poc_ok and rr_ok:
                    # TRADE VALID!
                    tp1 = close - (risk * TP1_RR)
                    tp2 = tp
                    risk_amount = account_balance * RISK_PERCENT
                    
                    result['event'] = "TRADE_SHORT"
                    result['event_details'] = f"Entry={close} SL={sl} TP1={tp1:.2f} TP2={tp2:.2f} RR={actual_rr:.2f}"
                    
                    success = place_market_order(mt5_symbol, "SELL", close, sl, tp1, tp2, risk_amount, config['tick_size'])
                    if success:
                        asset_state.last_trade_candle_ts = candle_ts
                else:
                    # Entry conditions not met
                    if result['event'] is None:
                        reasons = []
                        if risk <= 0: reasons.append(f"risk={risk:.2f}")
                        if tp < val: reasons.append(f"TP<VAL")
                        if not poc_ok: reasons.append(f"close<POC")
                        if not rr_ok: reasons.append(f"RR={actual_rr:.2f}<{MIN_RR}")
                        result['event'] = "FILTERED_ENTRY"
                        result['event_details'] = ", ".join(reasons)
            
            # Retour à INSIDE (comme backtest)
            asset_state.state = "INSIDE"
        else:
            # Toujours en breakout
            result['event'] = "STILL_BREAKOUT_UP"
            result['event_details'] = f"swing={asset_state.swing_extreme}"
    
    elif state == "BREAKOUT_DOWN":
        # Mettre à jour swing_extreme (comme backtest: min(swing_extreme, low))
        asset_state.swing_extreme = min(asset_state.swing_extreme, low)
        result['swing_extreme'] = asset_state.swing_extreme
        
        if close > val:
            # Failed breakout DOWN - évaluer LONG (comme backtest)
            breakout_duration_min = (candle_dt - asset_state.breakout_time).total_seconds() / 60.0
            
            # Flags pour les filtres (comme backtest)
            duration_ok = True
            poc_strength_ok = True
            vp_shape_ok = True
            
            # Filtre duration
            if USE_BREAKOUT_DURATION_FILTER:
                if breakout_duration_min >= MAX_BREAKOUT_DURATION_MINUTES:
                    duration_ok = False
                    result['event'] = "FILTERED_DURATION"
                    result['event_details'] = f"{breakout_duration_min:.1f}min >= {MAX_BREAKOUT_DURATION_MINUTES}min"
            
            # Filtre POC strength
            if USE_VP_STRUCTURE_FILTER:
                if poc_strength is None or poc_strength < MIN_POC_STRENGTH:
                    poc_strength_ok = False
                    if duration_ok:
                        result['event'] = "FILTERED_POC_STRENGTH"
                        result['event_details'] = f"{poc_strength:.2f}x < {MIN_POC_STRENGTH}x"
            
            # Filtre VP shape
            if USE_VP_SHAPE_FILTER:
                if vp_shape in EXCLUDED_VP_SHAPES:
                    vp_shape_ok = False
                    if duration_ok and poc_strength_ok:
                        result['event'] = "FILTERED_VP_SHAPE"
                        result['event_details'] = vp_shape
            
            # Tous les filtres OK + allow_long (comme backtest)
            if duration_ok and poc_strength_ok and vp_shape_ok and config['allow_long']:
                sl = asset_state.swing_extreme - config['sl_offset']
                risk = close - sl
                
                if TP_MODE == "POC":
                    tp = poc
                    actual_rr = (tp - close) / risk if risk > 0 else 0
                    # Filtre RR aberrant (comme backtest: > 10)
                    if actual_rr > 10:
                        result['event'] = "FILTERED_RR_ABERRANT"
                        result['event_details'] = f"RR={actual_rr:.1f}"
                        asset_state.state = "INSIDE"
                        result['state_after'] = "INSIDE"
                        return result
                else:
                    tp = close + (risk * TARGET_RR)
                    actual_rr = TARGET_RR
                
                poc_ok = (close <= poc) if FILTER_ENTRY_VS_POC else True
                rr_ok = actual_rr >= MIN_RR
                
                if risk > 0 and tp <= vah and poc_ok and rr_ok:
                    # TRADE VALID!
                    tp1 = close + (risk * TP1_RR)
                    tp2 = tp
                    risk_amount = account_balance * RISK_PERCENT
                    
                    result['event'] = "TRADE_LONG"
                    result['event_details'] = f"Entry={close} SL={sl} TP1={tp1:.2f} TP2={tp2:.2f} RR={actual_rr:.2f}"
                    
                    success = place_market_order(mt5_symbol, "BUY", close, sl, tp1, tp2, risk_amount, config['tick_size'])
                    if success:
                        asset_state.last_trade_candle_ts = candle_ts
                else:
                    # Entry conditions not met
                    if result['event'] is None:
                        reasons = []
                        if risk <= 0: reasons.append(f"risk={risk:.2f}")
                        if tp > vah: reasons.append(f"TP>VAH")
                        if not poc_ok: reasons.append(f"close>POC")
                        if not rr_ok: reasons.append(f"RR={actual_rr:.2f}<{MIN_RR}")
                        result['event'] = "FILTERED_ENTRY"
                        result['event_details'] = ", ".join(reasons)
            
            # Retour à INSIDE (comme backtest)
            asset_state.state = "INSIDE"
        else:
            # Toujours en breakout
            result['event'] = "STILL_BREAKOUT_DOWN"
            result['event_details'] = f"swing={asset_state.swing_extreme}"
    
    result['state_after'] = asset_state.state
    return result


def log_candle_info(result: dict, balance: float, equity: float):
    """Log les infos importantes (pas INSIDE, OUT_OF_SESSION, STILL_BREAKOUT)"""
    if not result['new_candle']:
        return
    
    event = result['event'] or "NONE"
    
    # Ne pas logger ces événements
    if event in ["INSIDE", "OUT_OF_SESSION", "STILL_BREAKOUT_UP", "STILL_BREAKOUT_DOWN"]:
        return
    
    symbol = result['symbol']
    dt = result['candle_dt'].strftime('%Y-%m-%d %H:%M') if result['candle_dt'] else "N/A"
    session = result['session'] or "N/A"
    
    logger.info(f"{'─' * 80}")
    logger.info(f"[{symbol}] 🕐 {dt} UTC | Session: {session} | Balance: ${balance:,.2f} | Equity: ${equity:,.2f}")
    
    if result['close']:
        logger.info(f"[{symbol}] Price: Close={result['close']:.2f} | High={result['high']:.2f} | Low={result['low']:.2f}")
    
    if result['poc']:
        logger.info(f"[{symbol}] VP: POC={result['poc']:.2f} | VAH={result['vah']:.2f} | VAL={result['val']:.2f} | Strength={result['poc_strength']:.2f}x | Shape={result['vp_shape']}")
    
    state_change = ""
    if result['state_before'] != result['state_after']:
        state_change = f" → {result['state_after']}"
    logger.info(f"[{symbol}] State: {result['state_before']}{state_change} | Swing: {result['swing_extreme']:.2f}")
    
    details = result['event_details'] or ""
    
    if event.startswith("TRADE"):
        emoji = "🎯"
    elif event.startswith("BREAKOUT"):
        emoji = "⚡"
    elif event.startswith("FILTERED"):
        emoji = "🚫"
    elif event == "SESSION_START":
        emoji = "🔄"
    elif event == "POSITION_ACTIVE":
        emoji = "📍"
    elif event == "NO_VP":
        emoji = "⚠️"
    else:
        emoji = "ℹ️"
    
    if details:
        logger.info(f"[{symbol}] {emoji} Event: {event} | {details}")
    else:
        logger.info(f"[{symbol}] {emoji} Event: {event}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("VP Failed Breakout - Live Trading")
    logger.info("ALIGNÉ SUR BACKTEST")
    logger.info("=" * 60)
    
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        mt5.shutdown()
        return
    
    logger.info(f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}")
    
    engine = get_pg_engine()
    
    enabled_assets = [a for a in ASSETS if a.get('enabled', False)]
    if not enabled_assets:
        logger.error("No enabled assets")
        mt5.shutdown()
        return
    
    asset_states = {}
    for config in enabled_assets:
        asset_states[config['symbol']] = AssetState(config)
        sessions = [s for s, v in config.get('sessions', {}).items() if v]
        logger.info(f"Asset: {config['symbol']} | Sessions: {sessions}")
    
    load_all_states(asset_states)
    
    logger.info(f"Filters: Duration<{MAX_BREAKOUT_DURATION_MINUTES}min | POC>{MIN_POC_STRENGTH}x | Exclude:{EXCLUDED_VP_SHAPES}")
    logger.info(f"Risk: {RISK_PERCENT*100}% | MIN_RR: {MIN_RR} | TP1_RR: {TP1_RR}")
    logger.info("=" * 60)
    
    try:
        while True:
            ref_asset = list(asset_states.values())[0] if asset_states else None
            db_time = None
            if ref_asset:
                ref_candles = fetch_candles_from_db(engine, ref_asset.config['candle_table'], limit=1)
                if ref_candles:
                    db_time = ref_candles[0]['dt']
            
            if db_time is None:
                logger.warning("No DB data available, waiting...")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue
            
            if daily_market_close_guard(db_time):
                time.sleep(60)
                continue
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Lost connection to MT5")
                time.sleep(10)
                continue
            
            balance = account_info.balance
            equity = account_info.equity
            
            for symbol, asset_state in asset_states.items():
                try:
                    manage_tp1_to_be(asset_state.mt5_symbol, asset_state.config['tick_size'])
                    result = detect_and_trade(asset_state, engine, balance)
                    log_candle_info(result, balance, equity)
                except Exception as e:
                    logger.error(f"[{symbol}] Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            save_all_states(asset_states)
            time.sleep(LOOP_INTERVAL_SECONDS)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        mt5.shutdown()
        logger.info("MT5 shutdown complete")


if __name__ == "__main__":
    main()