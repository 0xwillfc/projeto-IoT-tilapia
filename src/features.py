import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # cria colunas simples de tempo (hora, dia da semana e mes)
    out = df.copy()
    if "datetime" not in out.columns:
        return out

    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month_num"] = out["datetime"].dt.month
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lags: tuple[int, ...] = (1, 2, 3, 6),
    roll_windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    # cria defasagens (lags) e medias/desvios moveis para cada sensor
    # isso ajuda o modelo a "ver" tendencia recente dos sinais
    out = df.copy()
    for col in sensor_cols:
        if col not in out.columns:
            continue

        # exemplo: lag 1 = valor da linha anterior
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

        # rolling usa uma janela de valores passados
        for w in roll_windows:
            out[f"{col}_rollmean_{w}"] = out[col].shift(1).rolling(w).mean()
            out[f"{col}_rollstd_{w}"] = out[col].shift(1).rolling(w).std()

    return out


def make_supervised_table(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    # junta todas as features e remove linhas com nulos criados por lag/rolling
    out = add_time_features(df)
    out = add_lag_rolling_features(out, sensor_cols=sensor_cols)
    out = out.dropna(axis=0).reset_index(drop=True)
    return out
