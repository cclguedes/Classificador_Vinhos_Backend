import pytest
import json
from app import app


# 🔹 cliente de teste
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# 🔹 teste home
def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data


# 🔹 teste docs redirect
def test_docs(client):
    response = client.get('/docs')
    assert response.status_code == 302
    assert '/openapi' in response.location


# 🔹 teste predição válida
def test_predict(client):
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


# 🔹 teste erro (payload incompleto)
def test_predict_invalid(client):
    payload = {
        "fixed_acidity": 7.4
    }

    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 422