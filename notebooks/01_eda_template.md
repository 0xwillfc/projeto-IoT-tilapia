# EDA Template

## 1) Carregamento

```python
import pandas as pd
from src.data_loader import load_dataset

df = load_dataset("data/raw/Data Model IoTMLCQ 2024.xlsx")
df.head()
```

## 2) Qualidade de dados

```python
df.info()
df.isna().mean().sort_values(ascending=False)
```

## 3) Analise temporal

- Plot de `temperature_c`, `dissolved_oxygen_mg_l` e `ph` por tempo.
- Verificar periodicidade por hora (`hour`) e dia da semana (`dayofweek`).

## 4) Relacao com crescimento

- Scatter + correlacao de `average_fish_weight_g` vs sensores.
- Correlacao com defasagens (`lag`) de `O2`, `pH`, `temperatura`.

## 5) Relacao com risco

- Distribuicao de `health_status` ou `health_risk_bin`.
- Curvas de densidade por classe para `dissolved_oxygen_mg_l` e `temperature_c`.
