# Classificador de Vinhos - Back-end

Este projeto faz parte da minha avaliação na Pós-Graduação em Engenharia de Software da PUC-Rio. Trata-se de uma aplicação para classificar vinhos através de um modelo de ML, possuindo também uma API para integração com um frontend.

A API pode ser gerenciada pelo seguinte front-end: https://github.com/cclguedes/Classificador_Vinhos_Frontend

## Notebook

https://colab.research.google.com/drive/1N-43thTEkgqTTrm_fL83R-6JrMA5hnVu?usp=sharing

## Tecnologias utilizadas

- [Python](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Flask-OpenAPI3](https://pypi.org/project/flask-openapi3/)
- [Pydantic](https://docs.pydantic.dev/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Pytest](https://docs.pytest.org/)

## Instalação e execução

Clone o repositório:
```bash
git clone https://github.com/cclguedes/Classificador_Vinhos_Backend
```
Entre na pasta do projeto:
```bash
cd classificador_vinhos_backend
```
Crie e ative um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
```
```bash
source venv/bin/activate # Linux/macOS
```
```bash
venv\Scripts\activate # Windows
```
Instale as dependências:
```bash
pip install -r requirements.txt
```
Entre na pasta do backend:
```bash
cd backend
```
Execute a aplicação:
```bash
python app.py
```
## Sobre o autor

Sou Caio Guedes, engenheiro eletricista e especialista em gestão de projetos, trabalhando atualmente como Product Owner de projetos de tecnologia na indústria audiovisual.
https://www.linkedin.com/in/cclguedes/