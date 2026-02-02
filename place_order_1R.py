import MetaTrader5 as mt5
import sys
import os

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

RISK_PCT = 0.1    # Risque en %
FILENAME = "traded_symbols.txt"
SLIPPAGE_POINTS = 20 

# =============================================================================
# 1. OUTILS DE CALCUL
# =============================================================================

def _ccy_mid(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if not tick: tick = mt5.symbol_info_tick(symbol.upper())
    if not tick: return 0.0
    if tick.bid > 0 and tick.ask > 0:
        return (tick.bid + tick.ask) / 2.0
    return tick.last

def _fx_rate(ccy_from: str, ccy_to: str) -> float:
    ccy_from = ccy_from.upper()
    ccy_to = ccy_to.upper()
    if ccy_from == ccy_to: return 1.0
    rate = _ccy_mid(ccy_from + ccy_to)
    if rate > 0: return rate
    inv_rate = _ccy_mid(ccy_to + ccy_from)
    if inv_rate > 0: return 1.0 / inv_rate
    return 0.0

def calc_lots(symbol: str, risk_pct: float, entry: float, sl: float, balance: float, account_ccy: str, order_type: int) -> float:
    si = mt5.symbol_info(symbol)
    if not si: return 0.0
    risk_amount = balance * (risk_pct / 100.0)
    
    try:
        profit_per_lot = mt5.order_calc_profit(order_type, symbol, 1.0, entry, sl)
        risk_per_lot = abs(profit_per_lot) if profit_per_lot else 0.0
    except: risk_per_lot = 0.0
    
    if risk_per_lot <= 0:
        contract_size = si.trade_contract_size
        quote_ccy = symbol[-3:].upper() 
        loss_in_quote = abs(entry - sl) * contract_size
        if quote_ccy == account_ccy.upper(): risk_per_lot = loss_in_quote
        else:
            rate = _fx_rate(quote_ccy, account_ccy)
            if rate <= 0: return 0.0
            risk_per_lot = loss_in_quote * rate

    if risk_per_lot <= 0: return 0.0
    
    lots = risk_amount / risk_per_lot
    step = si.volume_step
    lots = round(lots / step) * step
    lots = max(si.volume_min, min(lots, si.volume_max))
    
    return float(f"{lots:.2f}")

# =============================================================================
# 2. SELECTION DU TICKER + OFFSET
# =============================================================================

def get_ticker_and_offset():
    """ Retourne (symbol, offset) """
    assets = []
    
    if os.path.exists(FILENAME):
        try:
            with open(FILENAME, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',')
                    ticker = parts[0].strip()
                    offset = 0.0
                    if len(parts) > 1:
                        try: offset = float(parts[1].strip())
                        except: pass
                    assets.append((ticker, offset))
        except: pass

    if assets:
        print(f"\n📋 --- SÉLECTION TICKER (RR 1:1 + OFFSET) ---")
        for i, (t, o) in enumerate(assets): 
            off_txt = f"(Off: {o})" if o != 0 else ""
            print(f"  {i+1}. {t} {off_txt}")
        print(f"  0. Manuel")
        
        while True:
            choice = input("\n👉 Choix : ").strip()
            if choice == '0': break 
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(assets):
                    if mt5.symbol_select(assets[idx][0], True): 
                        return assets[idx]
            print("❌ Invalide.")

    # Manuel
    while True:
        s = input("🔹 Ticker Manuel : ").strip()
        if mt5.symbol_select(s, True): 
            return s, 0.0
        print(f"   ❌ '{s}' introuvable.")

def get_float(prompt):
    while True:
        try: return float(input(prompt).replace(',', '.'))
        except: print("   ❌ Nombre invalide.")

# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    print(f"\n🚀 --- SETUP TRADE (AUTO RR 1:1 + OFFSET TV) ---")
    if not mt5.initialize(): return

    # 1. On récupère Ticker ET Offset du fichier
    ticker, offset = get_ticker_and_offset()
    
    if offset != 0:
        print(f"ℹ️  Offset actif : {offset:+.2f} (Broker = TV + Offset)")

    # 2. On demande le SL du TradingView
    tv_sl = get_float("🔹 Stop Loss (Valeur TV) : ")
    
    # 3. Conversion en SL Broker
    broker_sl = tv_sl + offset

    # --- 4. DÉTECTION SENS & CALCUL TP (ESTIMATIF) ---
    tick = mt5.symbol_info_tick(ticker)
    if not tick: return
    
    mid_price = (tick.bid + tick.ask) / 2.0

    # On compare le SL Broker au Prix Broker
    if broker_sl < mid_price:
        # SL en dessous = BUY
        order_type = mt5.ORDER_TYPE_BUY
        est_price = tick.ask
        dist = est_price - broker_sl
        tp = est_price + dist # RR 1:1
        side = "BUY"
    else:
        # SL au dessus = SELL
        order_type = mt5.ORDER_TYPE_SELL
        est_price = tick.bid
        dist = broker_sl - est_price
        tp = est_price - dist # RR 1:1
        side = "SELL"

    acc = mt5.account_info()
    est_lots = calc_lots(ticker, RISK_PCT, est_price, broker_sl, acc.equity, acc.currency, order_type)

    print("\n" + "="*40)
    print(f"📊 RÉSUMÉ (RR 1:1) : {ticker} ({side})")
    print(f"Prix Broker : {est_price} (Actuel)")
    print(f"SL Broker   : {broker_sl} (TV: {tv_sl})")
    print(f"TP Broker   : {tp:.5f} (Auto)")
    print(f"Lots        : {est_lots}")
    print("="*40)

    if est_lots <= 0: 
        print("❌ Erreur Lots=0 (Trop proche ou fonds insuffisants).")
        mt5.shutdown(); return

    # --- 5. EXECUTION REELLE ---
    if input("\n✅ FIRE ? (y/n) : ").lower() in ['y', 'o', 'oui', '']:
        
        # A. Mise à jour ultime du prix
        tick_final = mt5.symbol_info_tick(ticker)
        
        if order_type == mt5.ORDER_TYPE_BUY:
            final_price = tick_final.ask
            # B. Recalcul TP pour RR 1:1 EXACT sur le prix réel
            final_dist = final_price - broker_sl
            final_tp = final_price + final_dist
        else:
            final_price = tick_final.bid
            # B. Recalcul TP pour RR 1:1 EXACT sur le prix réel
            final_dist = broker_sl - final_price
            final_tp = final_price - final_dist

        # C. Recalcul Lots sur le prix réel
        final_lots = calc_lots(ticker, RISK_PCT, final_price, broker_sl, acc.equity, acc.currency, order_type)
        
        if final_lots <= 0:
            print("❌ Erreur Lots recalculés = 0.")
            mt5.shutdown(); return

        print(f"🔄 Recalcul : Prix {est_price} -> {final_price} | TP {tp:.2f} -> {final_tp:.2f}")

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": ticker,
            "volume": final_lots,
            "type": order_type,
            "price": final_price, 
            "sl": broker_sl,
            "tp": final_tp, 
            "deviation": SLIPPAGE_POINTS,
            "magic": 111000,
            "comment": "Python_RR1_Offset",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n🎉 ORDRE ENVOYÉ ! ({res.price}) | TP fixé à {final_tp:.5f}")
        else:
            print(f"\n💀 ERREUR: {res.comment} ({res.retcode})")
    else:
        print("\n🚫 Annulé.")

    mt5.shutdown()

if __name__ == "__main__":
    main()