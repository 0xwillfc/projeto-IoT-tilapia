import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_dataset
from features import make_supervised_table


def mape(y_true, y_pred):
    # erro percentual medio (em %)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0


def time_split(df, test_size=0.2):
    # split temporal: inicio para treino, fim para teste
    # evita misturar futuro no treino
    n = len(df)
    cut = int(n * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main():
    # 1) argumentos de linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Caminho para .xlsx/.csv")
    parser.add_argument("--outdir", default="outputs", help="Pasta de saida")
    args = parser.parse_args()

    # 2) carrega dados brutos
    raw = load_dataset(args.data)

    # 3) define alvo da regressao
    target = "average_fish_weight_g"
    if target not in raw.columns:
        raise ValueError(f"Coluna alvo ausente: {target}")

    # 4) cria features a partir dos sensores
    sensor_cols = ["temperature_c", "dissolved_oxygen_mg_l", "ph", "turbidity_ntu"]
    df = make_supervised_table(raw, sensor_cols=sensor_cols)

    # remove colunas que nao vamos usar no treino
    keep = [c for c in df.columns if c != target and c != "datetime"]
    model_df = df[keep + [target]].copy()

    # 5) separa treino e teste
    train_df, test_df = time_split(model_df, test_size=0.2)
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # 6) separa colunas numericas e categoricas
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # 7) pipeline de preprocessamento
    # numericas: preencher nulo + escalar
    # categoricas: preencher nulo + one hot encoding
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    # 8) modelo base de regressao
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # 9) predicao e metricas
    preds = pipe.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mape_pct": float(mape(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # 10) salva modelo e metricas
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "regression_model.joblib")
    with open(outdir / "regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
