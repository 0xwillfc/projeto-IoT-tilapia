import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_dataset
from features import make_supervised_table


def time_split(df, test_size=0.2):
    # split temporal: inicio treina, final testa
    n = len(df)
    cut = int(n * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _to_binary_health(df):
    # tenta montar a coluna de risco (0/1) a partir das colunas existentes
    if "health_status" in df.columns:
        s = df["health_status"].astype(str).str.strip().str.lower()
        return s.map(lambda x: 1 if x in {"at risk", "risk", "critical"} else 0)
    if "low_oxygen_alert" in df.columns:
        s = df["low_oxygen_alert"].astype(str).str.strip().str.lower()
        return s.map(lambda x: 1 if x in {"critical", "alert", "at risk"} else 0)

    # fallback simples se o dataset nao tiver label pronta:
    # risco = 1 quando o2 baixo ou temperatura alta
    do_col = "dissolved_oxygen_mg_l"
    temp_col = "temperature_c"
    risk = np.zeros(len(df), dtype=int)
    if do_col in df.columns:
        risk = np.where(df[do_col].astype(float) < 5.0, 1, risk)
    if temp_col in df.columns:
        risk = np.where(df[temp_col].astype(float) > 32.0, 1, risk)
    return risk


def main():
    # 1) argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Caminho para .xlsx/.csv")
    parser.add_argument("--outdir", default="outputs", help="Pasta de saida")
    args = parser.parse_args()

    # 2) carrega dados e cria alvo binario de risco
    raw = load_dataset(args.data)
    raw["health_risk_bin"] = _to_binary_health(raw)

    # 3) cria features dos sensores
    sensor_cols = ["temperature_c", "dissolved_oxygen_mg_l", "ph", "turbidity_ntu"]
    df = make_supervised_table(raw, sensor_cols=sensor_cols)

    # remove colunas que podem "vazar" informacao da label
    target = "health_risk_bin"
    keep = [c for c in df.columns if c not in {target, "datetime", "health_status", "low_oxygen_alert"}]
    model_df = df[keep + [target]].copy()

    # 4) separa treino e teste
    train_df, test_df = time_split(model_df, test_size=0.2)
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    # 5) identifica colunas numericas e categoricas
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # 6) preprocessamento (igual ao da regressao)
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

    # 7) modelo base de classificacao
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # 8) predicao e metricas
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="binary")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    if prob is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, prob))

    # 9) salva arquivos de saida
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "classification_model.joblib")
    with open(outdir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
