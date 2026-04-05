from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar modelo e scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "API de Classificação de Vinhos está rodando 🚀"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validar entrada
        required_fields = [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Campo obrigatório ausente: {field}"
                }), 400

        # Montar vetor na ordem correta
        values = [
            data["fixed_acidity"],
            data["volatile_acidity"],
            data["citric_acid"],
            data["residual_sugar"],
            data["chlorides"],
            data["free_sulfur_dioxide"],
            data["total_sulfur_dioxide"],
            data["density"],
            data["pH"],
            data["sulphates"],
            data["alcohol"]
        ]

        # Converter para numpy
        values = np.array([values])

        # Aplicar scaler
        values_scaled = scaler.transform(values)

        # Predição
        prediction = model.predict(values_scaled)

        return jsonify({
            "categoria": prediction[0]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)