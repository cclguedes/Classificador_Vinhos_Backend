from flask_openapi3 import OpenAPI, Info, Tag
from flask import redirect, jsonify
from flask_cors import CORS
from pydantic import BaseModel

import joblib
import numpy as np


# Configuração das informações básicas da API (título e versão)
info = Info(title="API Classificador de Vinhos", version="1.0.0")
app = OpenAPI(__name__, info=info)

# Habilitação de CORS para permitir requisições externas (ex.: frontend)
CORS(app, resources={r"/*": {"origins": "*"}})


# Definição de tags utilizadas na documentação da API
home_tag = Tag(
    name="Documentação",
    description="Acesso à documentação da API"
)

vinho_tag = Tag(
    name="Vinho",
    description="Classificação de qualidade de vinhos"
)


# Carregamento do modelo treinado e do scaler previamente salvos
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Definição do schema de entrada utilizado para validação dos dados via OpenAPI
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


# Rota raiz que redireciona automaticamente para a documentação da API
@app.get("/", tags=[home_tag])
def home():
    return redirect("/openapi")


# Endpoint responsável pela predição da qualidade do vinho
@app.post("/predict", tags=[vinho_tag])
def predict(body: VinhoInput):
    try:
        # Extração dos valores recebidos no corpo da requisição
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

        # Conversão para array numpy e aplicação do scaler
        values = np.array([values])
        values_scaled = scaler.transform(values)

        # Realização da predição com o modelo treinado
        prediction = model.predict(values_scaled)

        # Retorno da resposta no formato JSON
        return jsonify({
            "categoria": prediction[0]
        })

    except Exception as e:
        # Tratamento de erro genérico
        return jsonify({"error": str(e)}), 500


# Execução da aplicação em modo de desenvolvimento
if __name__ == "__main__":
    app.run(debug=True)