from flask_openapi3 import OpenAPI, Info, Tag
from flask import redirect, jsonify
from flask_cors import CORS
from pydantic import BaseModel

import pickle
import numpy as np
import pandas as pd


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


# Carregamento do pipeline treinado (inclui pré-processamento + modelo)
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Nomes das colunas na mesma ordem usada no treinamento
FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]


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
        # Montagem do DataFrame com os nomes de colunas usados no treinamento
        values = pd.DataFrame([[
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
        ]], columns=FEATURE_COLUMNS)

        # O pipeline aplica o pré-processamento e realiza a predição
        prediction = pipeline.predict(values)

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