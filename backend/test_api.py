import pytest
import json
from app import app


@pytest.fixture
def client():
    """
    Fixture responsável por inicializar o cliente de testes da aplicação Flask.
    Ativa o modo de teste e fornece uma instância reutilizável para os testes.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    """
    Verifica se a rota raiz (/) realiza corretamente o redirecionamento
    para a documentação da API (/openapi).
    """
    response = client.get('/')
    assert response.status_code == 302
    assert '/openapi' in response.location


def test_predict(client):
    """
    Testa o endpoint de predição com um payload válido,
    verificando se a resposta contém a categoria prevista
    dentro do conjunto esperado ("ruim" ou "bom").
    """
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 200

    data = json.loads(response.data)

    assert 'categoria' in data
    assert data['categoria'] in ['ruim', 'bom']


def test_predict_invalid(client):
    """
    Testa o comportamento do endpoint de predição quando recebe
    um payload incompleto, esperando retorno de erro de validação (HTTP 422).
    """
    payload = {
        "fixed_acidity": 7.4
    }

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 422
