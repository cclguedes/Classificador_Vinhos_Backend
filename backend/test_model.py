import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --- Configurações globais ---
MODEL_PATH = "model.pkl"
DATASET_URL = "https://raw.githubusercontent.com/cclguedes/Classificador_Vinhos_Backend/refs/heads/main/dataset/WineQT.csv"

# Threshold mínimo de desempenho aceito para o modelo ir para produção
ACCURACY_THRESHOLD = 0.70

# Nomes das colunas na mesma ordem usada no treinamento
FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]


def load_pipeline():
    """Carrega o pipeline (pré-processamento + modelo) a partir do arquivo .pkl."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def to_dataframe(values):
    """Converte uma lista ou array de amostras em DataFrame com os nomes de colunas corretos."""
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


def load_test_data():
    """
    Reproduz a mesma separação treino/teste feita no notebook,
    retornando apenas o conjunto de teste para avaliação.
    """
    dataset = pd.read_csv(DATASET_URL, delimiter=',')

    dataset["categoria"] = dataset["quality"].apply(
        lambda q: "ruim" if q <= 5 else "bom"
    )
    dataset = dataset.drop(columns=["quality"])

    X = dataset.drop(columns=["categoria", "Id"])
    y = dataset["categoria"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=7,
        stratify=y
    )

    return X_test, y_test


def test_pipeline_load():
    """
    Verifica se o arquivo do pipeline é carregado corretamente,
    garantindo que não está corrompido ou ausente.
    """
    pipeline = load_pipeline()
    assert pipeline is not None


def test_pipeline_prediction():
    """
    Testa a capacidade do pipeline de realizar uma predição válida
    a partir de uma única amostra de entrada, sem pré-processamento manual.
    """
    pipeline = load_pipeline()

    sample = to_dataframe([[7.4, 0.7, 0.0, 1.9, 0.076,
                            11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

    prediction = pipeline.predict(sample)

    assert prediction[0] in ["ruim", "bom"]


def test_pipeline_multiple_inputs():
    """
    Avalia o comportamento do pipeline ao receber múltiplas amostras,
    verificando se todas as previsões pertencem às classes esperadas.
    """
    pipeline = load_pipeline()

    samples = to_dataframe([
        [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
        [8.0, 0.5, 0.2, 2.5, 0.08,  15.0, 40.0, 0.9960, 3.30, 0.65, 10.5],
        [7.0, 0.3, 0.4, 2.0, 0.05,  10.0, 30.0, 0.9950, 3.40, 0.80, 12.5]
    ])

    predictions = pipeline.predict(samples)

    for pred in predictions:
        assert pred in ["ruim", "bom"]


def test_model_accuracy():
    """
    Avalia a acurácia do modelo no conjunto de teste e verifica se
    atende ao threshold mínimo de desempenho definido para produção.

    Threshold: acurácia >= 0.70

    Este teste impede a implantação de um modelo que não atenda
    ao requisito mínimo de qualidade, mesmo que tecnicamente funcional.
    """
    pipeline = load_pipeline()
    X_test, y_test = load_test_data()

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")
    print(f"Threshold mínimo exigido:      {ACCURACY_THRESHOLD:.4f}")

    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Acurácia {accuracy:.4f} abaixo do threshold mínimo de {ACCURACY_THRESHOLD:.4f}. "
        f"Modelo não aprovado para produção."
    )