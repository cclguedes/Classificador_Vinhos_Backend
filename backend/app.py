from flask_openapi3 import OpenAPI, Info, Tag
from flask import redirect, jsonify
from flask_cors import CORS
from pydantic import BaseModel

import joblib
import numpy as np


# 🔹 Info da API
info = Info(title="API Classificador de Vinhos", version="1.0.0")
app = OpenAPI(__name__, info=info)

# 🔥 CORS liberado pro front
CORS(app, resources={r"/*": {"origins": "*"}})


# 🔹 Tags (padrão faculdade)
home_tag = Tag(
    name="Documentação",
    description="Acesso à documentação da API"
)

vinho_tag = Tag(
    name="Vinho",
    description="Classificação de qualidade de vinhos"
)


# 🔹 Carregar modelo
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# 🔹 Schema (Swagger)
class VinhoInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# 🔹 Home
@app.get("/", tags=[home_tag])
def home():
    return {"message": "API de Classificação de Vinhos 🚀"}


# 🔹 Docs (redirect padrão)
@app.get("/docs", tags=[home_tag])
def docs():
    return redirect("/openapi")


# 🔹 Endpoint principal (CORRIGIDO 🔥)
@app.post("/predict", tags=[vinho_tag])
def predict(body: VinhoInput):
    try:
        values = [
            body.fixed_acidity,
            body.volatile_acidity,
            body.citric_acid,
            body.residual_sugar,
            body.chlorides,
            body.free_sulfur_dioxide,
            body.total_sulfur_dioxide,
            body.density,
            body.pH,
            body.sulphates,
            body.alcohol
        ]

        values = np.array([values])
        values_scaled = scaler.transform(values)

        prediction = model.predict(values_scaled)

        return jsonify({
            "categoria": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)