## Результаты экспериментов

Все результаты получены на одинаковой стратифицированной кросс-валидации: 5 фолдов, `shuffle=True`, `random_state=42`.

| Эксперимент | Модель | Признаки | CV accuracy | Std | Прирост |
|---|---|---|---:|---:|---:|
| Baseline | Logistic Regression | Исходные признаки | 0.795 | — | — |
| 1 | Logistic Regression | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | **0.832** | 0.015 | **+0.037** |
| 2 | Support Vector Machine | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | 0.829 | 0.017 | +0.034 |
| 3 | Decision Tree | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | 0.790 | 0.031 | −0.005 |\
| 4 | Random Forest | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | 0.815 | 0.015 | +0.020 |