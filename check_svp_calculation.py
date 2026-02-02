import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import sys
import warnings
warnings.filterwarnings('ignore')  # On ignore les warnings de connexion Pandas

load_dotenv()

# --- CONFIGURATION ---
SYMBOL = "XAUUSD"
CANDLE_TABLE = "candles_mt5_xauusd_1m" # Nom généré par ton script candle
TICK_TABLE = "market_ticks"
TICK_SIZE = 0.01  # Précision XAUUSD
VA_PERCENT = 0.70

# Plage de test (Session spécifique du 22 au 23 Janvier 2026)
TEST_START_STR = "2026-01-22 23:00:00" 
TEST_END_STR = "2026-01-23 08:00:00"

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('PG_HOST'), port=os.getenv('PG_PORT'),
            database=os.getenv('PG_DB'), user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"❌ Erreur DB: {e}")
        sys.exit(1)

def get_candles_stream(conn):
    """
    Récupère les bougies 1m pour la période de test.
    C'est notre 'horloge' pour le backtest.
    """
    # Conversion dates str -> timestamp ms (format de ta table candles)
    dt_start = datetime.strptime(TEST_START_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    dt_end = datetime.strptime(TEST_END_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    
    ts_start = int(dt_start.timestamp() * 1000)
    ts_end = int(dt_end.timestamp() * 1000)

    query = f"""
        SELECT ts, close 
        FROM {CANDLE_TABLE}
        WHERE ts >= {ts_start} AND ts <= {ts_end}
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, conn)
    
    # Conversion ts ms -> datetime UTC pour affichage
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df

def get_ticks_for_window(conn, dt_session_start, dt_current_candle):
    """
    Récupère les ticks précis entre le début de session et la bougie actuelle.
    """
    # Attention: Ta table market_ticks est en 'TIMESTAMP WITHOUT TIME ZONE' (UTC implicite)
    # On enlève le tzinfo pour matcher le format SQL si besoin, ou on passe des strings ISO.
    t_start = dt_session_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end = dt_current_candle.strftime("%Y-%m-%d %H:%M:%S")

    query = f"""
        SELECT last as price, volume
        FROM {TICK_TABLE}
        WHERE symbol = '{SYMBOL}'
        AND time >= '{t_start}' 
        AND time <= '{t_end}'
    """
    return pd.read_sql(query, conn)

def calculate_svp(df_ticks):
    if df_ticks.empty:
        return None, None, None, 0

    # 1. Binning
    df_ticks['price_bin'] = (df_ticks['price'] / TICK_SIZE).round() * TICK_SIZE
    
    # 2. GroupBy (Somme des volumes '1.0')
    profile = df_ticks.groupby('price_bin')['volume'].sum().reset_index()
    profile = profile.sort_values('price_bin').reset_index(drop=True)
    
    total_volume = profile['volume'].sum()
    target_volume = total_volume * VA_PERCENT
    
    # 3. POC
    poc_idx = profile['volume'].idxmax()
    poc_price = profile.iloc[poc_idx]['price_bin']
    poc_vol = profile.iloc[poc_idx]['volume']
    
    # 4. VA Expansion
    current_vol = poc_vol
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1
    max_idx = len(profile) - 1
    
    while current_vol < target_volume:
        vol_up = profile.iloc[up_idx]['volume'] if up_idx <= max_idx else 0
        vol_down = profile.iloc[down_idx]['volume'] if down_idx >= 0 else 0
        
        if vol_up == 0 and vol_down == 0:
            break
            
        if vol_up > vol_down:
            current_vol += vol_up
            up_idx += 1
        else:
            current_vol += vol_down
            down_idx -= 1
            
    vah_idx = min(up_idx - 1, max_idx)
    val_idx = max(down_idx + 1, 0)
    
    return poc_price, profile.iloc[vah_idx]['price_bin'], profile.iloc[val_idx]['price_bin'], total_volume

def main():
    conn = get_db_connection()
    
    print(f"⏳ Chargement des bougies 1m pour la simulation ({TEST_START_STR} -> {TEST_END_STR})...")
    df_candles = get_candles_stream(conn)
    
    if df_candles.empty:
        print("❌ Aucune bougie trouvée dans la table candles_mt5... Vérifie tes dates ou le nom de la table.")
        return

    print(f"✅ {len(df_candles)} minutes à rejouer. Démarrage du calcul SVP...")
    print("-" * 60)
    print(f"{'HEURE UTC':<20} | {'PRIX CLOSE':<10} | {'POC':<10} | {'VAH':<10} | {'VAL':<10} | {'TICKS'}")
    print("-" * 60)

    # DÉBUT DE SESSION FIXE (Pour cette simulation)
    # Dans la réalité, on calculerait ça dynamiquement. Ici on hardcode le 22/01 23h00.
    session_start_dt = datetime.strptime(TEST_START_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    for row in df_candles.itertuples():
        current_dt = row.dt
        current_close = row.close

        # 1. On interroge les ticks accumulés depuis 23h00 jusqu'à maintenant
        df_ticks = get_ticks_for_window(conn, session_start_dt, current_dt)
        
        # 2. Calcul du SVP
        poc, vah, val, vol = calculate_svp(df_ticks)
        
        if poc is None:
            continue

        # 3. Affichage
        print(f"{current_dt.strftime('%d-%m %H:%M')}        | {current_close:<10.2f} | {poc:<10.2f} | {vah:<10.2f} | {val:<10.2f} | {int(vol)}")

    conn.close()

if __name__ == "__main__":
    main()