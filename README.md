## Результаты экспериментов

Все результаты получены на одинаковой стратифицированной кросс-валидации: 5 фолдов, `shuffle=True`, `random_state=42`.

| Эксперимент | Модель | Признаки | Основные параметры | CV accuracy | Std | Прирост |
|---|---|---|---|---:|---:|---:|
| Baseline | Logistic Regression | Исходные признаки | `C=1.0` | 0.795 | — | — |
| 1.1 | Logistic Regression | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `C=1.0` | 0.832 | 0.015 | +0.037 |
| 1.2 | Logistic Regression | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `C=0.1` | **0.832** | **0.008** | **+0.037** |
| 2.1 | SVC | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | RBF, базовые параметры | 0.829 | 0.017 | +0.034 |
| 3.1 | Decision Tree | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | Базовые параметры | 0.790 | 0.031 | -0.005 |
| 4.1 | Random Forest | Исходные + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `n_estimators=500` | 0.815 | 0.015 | +0.020 |

Подбор регуляризации не увеличил среднюю accuracy Logistic Regression, но снизил разброс между фолдами с `0.015` до `0.008`. Поэтому для дальнейшего сравнения фиксируем `C=0.1`.