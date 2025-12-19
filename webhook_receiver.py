from flask import Flask, request, jsonify

app = Flask(__name__)
PORT_LOCAL = 5000

@app.route('/webhook/tradingview', methods=['POST'])
def handle_webhook():
    """
    Point de terminaison qui reçoit les requêtes POST de TradingView.
    """
    try:
        # Tente de récupérer les données envoyées au format JSON
        data = request.json
        
        # Affiche le contenu dans le terminal
        print("--- Webhook Reçu ---")
        print("Contenu JSON :")
        print(data)
        print("--------------------")
        
        # Renvoie une réponse HTTP 200 (OK) à TradingView pour confirmer la réception
        return jsonify({'status': 'OK', 'message': 'Webhook reçu et affiché.'}), 200

    except Exception as e:
        # En cas d'erreur de parsing ou autre
        print(f"Erreur lors du traitement du webhook: {e}")
        return jsonify({'status': 'ERROR', 'message': str(e)}), 400

if __name__ == '__main__':
    print(f"Démarrage du serveur Flask. Écoute sur le port {PORT_LOCAL}...")
    print("Lancez ngrok dans un autre terminal pour exposer ce port !")
    app.run(port=PORT_LOCAL)