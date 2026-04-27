import re
from pathlib import Path

import pandas as pd


def _normalize_token(text: str) -> str:
    # deixa o nome da coluna em um formato padrao para facilitar o mapeamento
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


ALIASES = {
    "datetime": "datetime",
    "date": "datetime",
    "month": "month",
    "averagefishweightg": "average_fish_weight_g",
    "averagefishweight": "average_fish_weight_g",
    "survivalrate": "survival_rate_pct",
    "survivalratepct": "survival_rate_pct",
    "diseaseoccurrencecases": "disease_occurrence_cases",
    "diseaseoccurrence": "disease_occurrence_cases",
    "temperaturec": "temperature_c",
    "temperature": "temperature_c",
    "dissolvedoxygenmgl": "dissolved_oxygen_mg_l",
    "dissolvedoxygen": "dissolved_oxygen_mg_l",
    "do": "dissolved_oxygen_mg_l",
    "ph": "ph",
    "turbidityntu": "turbidity_ntu",
    "turbidity": "turbidity_ntu",
    "oxygenationautomatic": "oxygenation_automatic",
    "oxygenationinterventions": "oxygenation_interventions",
    "correctiveinterventions": "corrective_interventions",
    "thermalriskindex": "thermal_risk_index",
    "lowoxygenalert": "low_oxygen_alert",
    "healthstatus": "health_status",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # troca nomes "originais" por nomes curtos e padronizados
    rename_map = {}
    for col in df.columns:
        token = _normalize_token(col)
        rename_map[col] = ALIASES.get(token, token)
    out = df.rename(columns=rename_map).copy()
    return out


def _normalize_yes_no(val):
    # normaliza varias formas de sim/nao para "yes" ou "no"
    if pd.isna(val):
        return val
    text = str(val).strip().lower()
    if text in {"yes", "y", "sim", "true", "1"}:
        return "yes"
    if text in {"no", "n", "nao", "não", "false", "0"}:
        return "no"
    return text


def load_dataset(path: str | Path) -> pd.DataFrame:
    # funcao principal para carregar o arquivo e devolver um dataframe pronto
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    # aceita excel ou csv
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato nao suportado: {path.suffix}")

    # padroniza nome de colunas para facilitar o restante do pipeline
    df = standardize_columns(df)

    # se existir coluna de data/hora, converte e ordena
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime")

    # normaliza respostas yes/no em colunas de intervencao
    for yn_col in ["oxygenation_automatic", "oxygenation_interventions"]:
        if yn_col in df.columns:
            df[yn_col] = df[yn_col].apply(_normalize_yes_no)

    return df
