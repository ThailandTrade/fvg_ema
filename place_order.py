import MetaTrader5 as mt5
import sys

# =============================================================================
# 1. OUTILS DE CALCUL (MATHS & CONVERSION)
# =============================================================================

def _ccy_mid(symbol: str) -> float:
    # On garde la casse telle quelle pour le tick, au cas où le broker est strict
    tick = mt5.symbol_info_tick(symbol)
    if not tick: 
        # Si échec, on tente en majuscule par sécurité pour les paires Forex standard
        tick = mt5.symbol_info_tick(symbol.upper())
    
    if not tick: return 0.0
    
    if tick.bid > 0 and tick.ask > 0:
        return (tick.bid + tick.ask) / 2.0
    return tick.last

def _fx_rate(ccy_from: str, ccy_to: str) -> float:
    # Les codes devises (EUR, USD) sont toujours standards en majuscules pour le calcul
    ccy_from = ccy_from.upper()
    ccy_to = ccy_to.upper()
    
    if ccy_from == ccy_to: return 1.0
    
    # 1. Direct (ex: EURUSD)
    rate = _ccy_mid(ccy_from + ccy_to)
    if rate > 0: return rate
    
    # 2. Inverse (ex: USDEUR n'existe pas -> 1/EURUSD)
    inv_rate = _ccy_mid(ccy_to + ccy_from)
    if inv_rate > 0: return 1.0 / inv_rate
    
    return 0.0

def calc_lots(symbol: str, risk_pct: float, entry: float, sl: float, balance: float, account_ccy: str) -> float:
    # On récupère les infos du symbole (en respectant la casse fournie)
    si = mt5.symbol_info(symbol)
    if not si: return 0.0

    risk_amount = balance * (risk_pct / 100.0)
    
    # Tentative 1 : Fonction native MT5
    order_type = mt5.ORDER_TYPE_BUY if entry > sl else mt5.ORDER_TYPE_SELL
    try:
        profit_per_lot = mt5.order_calc_profit(order_type, symbol, 1.0, entry, sl)
        risk_per_lot = abs(profit_per_lot) if profit_per_lot else 0.0
    except Exception:
        risk_per_lot = 0.0
    
    # Tentative 2 : Calcul manuel (Fallback)
    if risk_per_lot <= 0:
        contract_size = si.trade_contract_size
        
        # Extraction de la devise de cotation (Quote Currency)
        # On suppose que les 3 derniers caractères sont la devise (ex: USD dans EURUSD)
        # On force .upper() ICI pour que le calcul de taux de change fonctionne
        # même si le ticker est "eurusd"
        quote_ccy = symbol[-3:].upper() 
        
        loss_in_quote = abs(entry - sl) * contract_size
        
        # Conversion vers la devise du compte
        if quote_ccy == account_ccy.upper():
            risk_per_lot = loss_in_quote
        else:
            rate = _fx_rate(quote_ccy, account_ccy)
            if rate <= 0: 
                print(f"   [WARN] Taux de change introuvable pour {quote_ccy}->{account_ccy}")
                return 0.0
            risk_per_lot = loss_in_quote * rate

    if risk_per_lot <= 0: return 0.0

    lots = risk_amount / risk_per_lot
    
    # Normalisation aux steps du broker
    step = si.volume_step
    lots = round(lots / step) * step
    lots = max(si.volume_min, min(lots, si.volume_max))
    
    return float(f"{lots:.2f}")

# =============================================================================
# 2. INPUTS (MODIFIÉ: Sensibilité à la casse)
# =============================================================================

def get_valid_ticker():
    while True:
        # MODIFICATION : On ne force plus le .upper()
        # On fait juste un .strip() pour éviter les espaces accidentels
        s = input("🔹 Ticker (ex: EURUSD, eurusd, BTCUSD.pro...) : ").strip()
        
        if mt5.symbol_select(s, True):
            return s
        print(f"   ❌ Le ticker '{s}' est introuvable sur MT5. Vérifie la casse exacte.")

def get_float(prompt):
    while True:
        try:
            val = input(prompt).replace(',', '.')
            return float(val)
        except ValueError:
            print("   ❌ Ce n'est pas un nombre valide.")

# =============================================================================
# 3. PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("\n🚀 --- SETUP DU TRADE (Interactive) ---")
    
    if not mt5.initialize():
        print("❌ Erreur connexion MT5")
        return

    # 1. Inputs
    ticker = get_valid_ticker()
    entry  = get_float(f"🔹 Entry Price pour {ticker} : ")
    sl     = get_float("🔹 Stop Loss (SL) : ")
    tp     = get_float("🔹 Take Profit (TP) : ")
    risk   = get_float("🔹 Risque (%) : ")

    # 2. Infos Compte
    acc = mt5.account_info()
    balance = acc.equity 
    currency = acc.currency
    
    tick = mt5.symbol_info_tick(ticker)
    current_bid = tick.bid
    current_ask = tick.ask

    # 3. Logique Direction (Limit/Stop/Buy/Sell)
    direction = ""
    order_type = None
    order_str = ""

    if sl < entry:
        # BUY
        if entry > current_ask:
            order_type = mt5.ORDER_TYPE_BUY_STOP
            order_str = "BUY STOP"
        else:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
            order_str = "BUY LIMIT"
    else:
        # SELL
        if entry < current_bid:
            order_type = mt5.ORDER_TYPE_SELL_STOP
            order_str = "SELL STOP"
        else:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT
            order_str = "SELL LIMIT"

    # 4. Calcul Lot
    lots = calc_lots(ticker, risk, entry, sl, balance, currency)
    
    risk_dist = abs(entry - sl)
    reward_dist = abs(tp - entry)
    rr = reward_dist / risk_dist if risk_dist > 0 else 0

    # 5. Confirmation
    print("\n" + "="*40)
    print(f"📊 RÉSUMÉ : {ticker}")
    print("="*40)
    print(f"Ordre      : {order_str}")
    print(f"Prix Entry : {entry}")
    print(f"SL         : {sl}")
    print(f"TP         : {tp}")
    print("-" * 40)
    print(f"Capital    : {balance:.2f} {currency}")
    print(f"Risque     : {risk}%")
    print(f"LOT SIZE   : {lots} lots")
    print(f"R:R        : 1:{rr:.2f}")
    print("="*40)

    if lots <= 0:
        print("❌ Erreur de calcul de lot (Résultat 0). Vérifie le SL ou le Ticker.")
        mt5.shutdown()
        return

    confirm = input("\n✅ GO ? (y/n) : ").lower()
    
    if confirm in ['y', 'yes', 'o', 'oui']:
        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": ticker,
            "volume": lots,
            "type": order_type,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 999000,
            "comment": "InteractiveScript",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n🎉 SUCCÈS ! Ticket: {res.order}")
        else:
            print(f"\n💀 ÉCHEC. Code: {res.retcode}")
            print(f"Info: {res.comment}")
    else:
        print("\n🚫 Annulé.")

    mt5.shutdown()

if __name__ == "__main__":
    main()