# projeto iot + tilapia

esse projeto e um ponto de partida para estudar ml com dados reais de piscicultura.
ideia: usar pH, o2, temperatura e outros sinais do viveiro para entender crescimento e risco de saude do peixe.

dataset:
[kaggle - iot monitoring of water quality and tilapia](https://www.kaggle.com/datasets/jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia)

## o que esse projeto faz

1. treina um modelo de regressao para prever `average_fish_weight_g` (peso medio do peixe).
2. treina um modelo de classificacao para prever risco de saude (`health_risk_bin`).
3. salva modelo e metricas na pasta `outputs`.

## estrutura da pasta

```text
iot-tilapia-ml-draft/
  data/
    README.md
    raw/                              # coloque aqui o arquivo do kaggle (.xlsx)
  notebooks/
    01_eda_template.md
  outputs/                            # saida dos modelos e metricas
  src/
    data_loader.py                    # carrega e organiza o dataset
    features.py                       # cria novas colunas para o modelo
    train_regression.py               # treino para prever peso
    train_classification.py           # treino para prever risco
  requirements.txt
```

## passo a passo (bem simples)

1. criar ambiente virtual

```bash
python -m venv .venv
```

2. ativar ambiente (windows powershell)

```bash
.venv\Scripts\activate
```

3. instalar bibliotecas

```bash
pip install -r requirements.txt
```

4. baixar o dataset e colocar o arquivo aqui:
`data/raw/Data Model IoTMLCQ 2024.xlsx`

## rodar treino de peso (regressao)

```bash
python src/train_regression.py --data "data/raw/Data Model IoTMLCQ 2024.xlsx"
```

## rodar treino de risco (classificacao)

```bash
python src/train_classification.py --data "data/raw/Data Model IoTMLCQ 2024.xlsx"
```

## arquivos que vao aparecer em outputs

- `outputs/regression_model.joblib`
- `outputs/regression_metrics.json`
- `outputs/classification_model.joblib`
- `outputs/classification_metrics.json`

## dicas de iniciante

1. primeiro rode os scripts sem mexer em nada, so para ver funcionar.
2. depois abra o `notebooks/01_eda_template.md` e faca analises simples dos dados.
