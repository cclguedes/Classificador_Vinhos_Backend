import pickle
import numpy as np


def test_model_load():
    """
    Verifica se o arquivo do modelo é carregado corretamente,
    garantindo que não está corrompido ou ausente.
    """
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    assert model is not None


def test_model_prediction():
    """
    Testa a capacidade do modelo de realizar uma predição válida
    a partir de uma única amostra de entrada.
    """
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = np.array([[7.4, 0.7, 0.0, 1.9, 0.076,
                        11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

    # CART não requer padronização — predição direta
    prediction = model.predict(sample)

    assert prediction[0] in ["ruim", "bom"]


def test_model_multiple_inputs():
    """
    Avalia o comportamento do modelo ao receber múltiplas amostras,
    verificando se todas as previsões retornadas pertencem às classes esperadas.
    """
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    samples = np.array([
        [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
        [8.0, 0.5, 0.2, 2.5, 0.08, 15.0, 40.0, 0.996, 3.3, 0.65, 10.5],
        [7.0, 0.3, 0.4, 2.0, 0.05, 10.0, 30.0, 0.995, 3.4, 0.8, 12.5]
    ])

    # CART não requer padronização — predição direta
    predictions = model.predict(samples)

    for pred in predictions:
        assert pred in ["ruim", "bom"]