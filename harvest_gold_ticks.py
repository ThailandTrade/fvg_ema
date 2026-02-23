import MetaTrader5 as mt5
import psycopg2
from psycopg2 import extras
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import time

# --- CONFIGURATION ---
load_dotenv()

# Symboles à surveiller : {symbole_mt5: nom_table_postgres}
SYMBOLS = {
    "XAUUSD": "market_ticks_xauusd",
    "XAGUSD": "market_ticks_xagusd",
}

DEFAULT_DAYS_BACK = 1000  # Nombre de jours à récupérer si DB vide
CHUNK_SIZE_HOURS = 24    # Taille des morceaux pour le téléchargement historique
SLEEP_INTERVAL = 1       # Pause en mode Live
UTC = timezone.utc

# --- LOGIQUE DE SYNCHRO TEMPORELLE (IDENTIQUE AU SCRIPT CANDLES) ---

def get_last_sunday(year: int, month: int) -> datetime:
    d = datetime(year, month, 1, tzinfo=UTC) + timedelta(days=31)
    if d.month != month:
        d -= timedelta(days=d.day)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def get_server_offset_hours(utc_ms: int) -> int:
    dt_utc = datetime.fromtimestamp(utc_ms / 1000, tz=UTC)
    year = dt_utc.year
    dst_start = get_last_sunday(year, 3) + timedelta(hours=1)
    dst_end   = get_last_sunday(year, 10) + timedelta(hours=1)
    return 3 if dst_start <= dt_utc < dst_end else 2

def utc_ms_to_server_ms(utc_ms: int) -> int:
    return utc_ms + get_server_offset_hours(utc_ms) * 3600 * 1000

def server_ms_to_utc_ms(server_ms: int) -> int:
    approx = server_ms - 2 * 3600 * 1000
    return server_ms - get_server_offset_hours(approx) * 3600 * 1000

# --- BASE DE DONNÉES ---

def get_db_connection():
    try:
        return psycopg2.connect(
            host=os.getenv('PG_HOST'),
            port=os.getenv('PG_PORT'),
            database=os.getenv('PG_DB'),
            user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD')
        )
    except Exception as e:
        print(f"❌ Erreur connexion DB: {e}")
        return None

def get_last_sync_time_ms(conn, table_name) -> int | None:
    """Récupère le timestamp (ms UTC) du dernier tick en base."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT MAX(ts) FROM {table_name}")
        res = cursor.fetchone()
        return int(res[0]) if res and res[0] else None
    except Exception as e:
        print(f"⚠️ [{table_name}] Impossible de lire le dernier tick: {e}")
        return None
    finally:
        cursor.close()

def init_db(conn):
    """Crée une table par symbole avec ts en ms UTC (comme candles)."""
    cursor = conn.cursor()
    
    for symbol, table_name in SYMBOLS.items():
        query_table = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ts BIGINT PRIMARY KEY,
                time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                bid DOUBLE PRECISION,
                ask DOUBLE PRECISION,
                last DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                flags INTEGER
            );
        """
        query_index = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name} (time);"
        
        cursor.execute(query_table)
        cursor.execute(query_index)
        print(f"📊 Table '{table_name}' prête.")
    
    conn.commit()
    cursor.close()

def insert_ticks(conn, ticks, table_name, last_ts_ms: int | None):
    """Insère un lot de ticks pour une table donnée."""
    if ticks is None or len(ticks) == 0:
        return 0, None
    
    cursor = conn.cursor()
    data_values = []

    for t in ticks:
        # MT5 retourne en server time
        raw_server_ms = int(t['time_msc']) if 'time_msc' in t.dtype.names else int(t['time']) * 1000
        
        # Convertir Server -> UTC (comme le script candles)
        ts_utc_ms = server_ms_to_utc_ms(raw_server_ms)
        
        # Filtre anti-doublon strict
        if last_ts_ms and ts_utc_ms <= last_ts_ms:
            continue

        # Timestamp human readable
        dt_utc = datetime.fromtimestamp(ts_utc_ms / 1000.0, tz=UTC).replace(tzinfo=None)
        
        price = t['last'] if t['last'] > 0 else t['bid']
        data_values.append((
            ts_utc_ms,
            dt_utc,
            float(t['bid']),
            float(t['ask']),
            float(price),
            1.0,
            int(t['flags'])
        ))
        
        last_ts_ms = ts_utc_ms

    if data_values:
        query = f"INSERT INTO {table_name} (ts, time, bid, ask, last, volume, flags) VALUES %s ON CONFLICT (ts) DO NOTHING"
        extras.execute_values(cursor, query, data_values)
        conn.commit()
        count = len(data_values)
        last_inserted_ms = data_values[-1][0]
    else:
        count = 0
        last_inserted_ms = None

    cursor.close()
    return count, last_inserted_ms

# --- GESTION MULTI-SYMBOLES ---

def sync_historical_for_symbol(conn, symbol, table_name, cursor_utc_ms: int) -> tuple[bool, int]:
    """
    Synchronise l'historique pour UN symbole depuis cursor_utc_ms.
    Retourne (done, new_cursor_utc_ms).
    """
    now_utc_ms = int(time.time() * 1000)
    
    # Si on est à moins de 1 minute du temps réel, historique rattrapé
    if (now_utc_ms - cursor_utc_ms) < 60_000:
        return True, cursor_utc_ms
    
    # Définir la fin du chunk (en UTC ms)
    chunk_end_utc_ms = cursor_utc_ms + (CHUNK_SIZE_HOURS * 3600 * 1000)
    if chunk_end_utc_ms > now_utc_ms:
        chunk_end_utc_ms = now_utc_ms

    # Pour affichage
    cursor_dt = datetime.fromtimestamp(cursor_utc_ms / 1000, tz=UTC)
    chunk_end_dt = datetime.fromtimestamp(chunk_end_utc_ms / 1000, tz=UTC)
    print(f"📥 [{symbol}] {cursor_dt.strftime('%Y-%m-%d %H:%M:%S')} -> {chunk_end_dt.strftime('%Y-%m-%d %H:%M:%S')} ... ", end="")

    # CONVERTIR UTC -> SERVER TIME avant d'appeler MT5 (comme le script candles)
    cursor_server_ms = utc_ms_to_server_ms(cursor_utc_ms)
    chunk_end_server_ms = utc_ms_to_server_ms(chunk_end_utc_ms)
    
    # MT5 attend des datetime naive en server time
    cursor_server_dt = datetime.fromtimestamp(cursor_server_ms / 1000).replace(tzinfo=None)
    chunk_end_server_dt = datetime.fromtimestamp(chunk_end_server_ms / 1000).replace(tzinfo=None)

    # Téléchargement MT5
    ticks = mt5.copy_ticks_range(symbol, cursor_server_dt, chunk_end_server_dt, mt5.COPY_TICKS_ALL)
    
    if ticks is not None and len(ticks) > 0:
        count, last_ts_ms = insert_ticks(conn, ticks, table_name, cursor_utc_ms)
        if count > 0:
            print(f"✅ {count} ticks")
            return False, last_ts_ms
        else:
            print("⚠️ doublons ignorés")
            return False, chunk_end_utc_ms
    else:
        print("∅ vide")
        return False, chunk_end_utc_ms

def sync_live_for_symbol(conn, symbol, table_name):
    """Récupère les nouveaux ticks en temps réel pour UN symbole."""
    last_ts_ms = get_last_sync_time_ms(conn, table_name)
    if not last_ts_ms:
        return
        
    now_utc_ms = int(time.time() * 1000)
    start_utc_ms = last_ts_ms + 1
    
    if (now_utc_ms - start_utc_ms) < 500:
        return

    # CONVERTIR UTC -> SERVER TIME avant d'appeler MT5
    start_server_ms = utc_ms_to_server_ms(start_utc_ms)
    now_server_ms = utc_ms_to_server_ms(now_utc_ms)
    
    # MT5 attend des datetime naive en server time
    start_server_dt = datetime.fromtimestamp(start_server_ms / 1000).replace(tzinfo=None)
    now_server_dt = datetime.fromtimestamp(now_server_ms / 1000).replace(tzinfo=None)

    ticks = mt5.copy_ticks_range(symbol, start_server_dt, now_server_dt, mt5.COPY_TICKS_ALL)
    
    if ticks is not None and len(ticks) > 0:
        count, last_inserted_ms = insert_ticks(conn, ticks, table_name, last_ts_ms)
        if count > 0:
            last_dt = datetime.fromtimestamp(last_inserted_ms / 1000, tz=UTC)
            print(f"⚡ [{symbol}] +{count} ticks (Dernier: {last_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")

# --- CŒUR DE LA RÉCOLTE MULTI-SYMBOLES ---

def harvest_ticks():
    if not mt5.initialize():
        print(f"❌ Erreur MT5: {mt5.last_error()}")
        return

    # Activer tous les symboles
    for symbol in SYMBOLS.keys():
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Symbole {symbol} introuvable ou impossible à activer.")
            return

    conn = get_db_connection()
    if not conn:
        return
    
    init_db(conn)

    print(f"🚀 Démarrage pour {len(SYMBOLS)} symboles : {', '.join(SYMBOLS.keys())}")

    # --- ÉTAPE 1 : INITIALISER LES CURSEURS (en ms UTC) ---
    now_utc_ms = int(time.time() * 1000)
    cursors = {}
    historical_done = {}

    for symbol, table_name in SYMBOLS.items():
        last_ts_ms = get_last_sync_time_ms(conn, table_name)
        if last_ts_ms:
            cursors[symbol] = last_ts_ms
            last_dt = datetime.fromtimestamp(last_ts_ms / 1000, tz=UTC)
            print(f"📅 [{symbol}] Reprise depuis : {last_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        else:
            cursors[symbol] = now_utc_ms - (DEFAULT_DAYS_BACK * 24 * 3600 * 1000)
            start_dt = datetime.fromtimestamp(cursors[symbol] / 1000, tz=UTC)
            print(f"📅 [{symbol}] Base vide, début : {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        historical_done[symbol] = False

    # --- ÉTAPE 2 : RATTRAPAGE HISTORIQUE (ROUND-ROBIN) ---
    print("⏳ Synchronisation historique (round-robin)...")
    
    while not all(historical_done.values()):
        for symbol, table_name in SYMBOLS.items():
            if historical_done[symbol]:
                continue
                
            done, new_cursor = sync_historical_for_symbol(conn, symbol, table_name, cursors[symbol])
            cursors[symbol] = new_cursor
            historical_done[symbol] = done
            
            if done:
                print(f"🏁 [{symbol}] Historique rattrapé.")
        
        time.sleep(0.1)

    print("✅ Historiques synchronisés. Passage en mode LIVE.")

    # --- ÉTAPE 3 : MODE LIVE ---
    print(f"🚀 Mode Live ({SLEEP_INTERVAL}s). Ctrl+C pour arrêter.")
    
    try:
        while True:
            for symbol, table_name in SYMBOLS.items():
                sync_live_for_symbol(conn, symbol, table_name)
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt du script.")
    finally:
        if conn:
            conn.close()
        mt5.shutdown()

if __name__ == "__main__":
    harvest_ticks()