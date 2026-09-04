## Результаты экспериментов

Все результаты получены на одинаковой стратифицированной кросс-валидации: 5 фолдов, `shuffle=True`, `random_state=42`. Прирост рассчитан относительно baseline с `CV accuracy = 0.795`.

| Эксперимент | Модель | Признаки | Основные параметры | CV accuracy | Std | Прирост |
|---|---|---|---|---:|---:|---:|
| Baseline | Logistic Regression | Исходные признаки | `C=1.0` | 0.795 | — | — |
| 1.1 | Logistic Regression | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `C=1.0` | 0.832 | 0.015 | +0.037 |
| 1.2 | Logistic Regression | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `C=0.1` | **0.832** | **0.008** | **+0.037** |
| 2.1 | SVC | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `kernel='rbf'`, `C=1.0`, `gamma='scale'` | 0.829 | 0.017 | +0.034 |
| 2.2 | SVC | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `kernel='rbf'`, `C=10`, `gamma=0.01` | 0.835 | 0.009 | +0.040 |
| 2.3 | SVC | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `kernel='rbf'`, `C=10`, `gamma=0.003` | **0.835** | **0.009** | **+0.040** |
| 3.1 | Decision Tree | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | Базовые параметры | **0.790** | **0.031** | **−0.005** |
| 4.1 | Random Forest | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `n_estimators=500` | 0.815 | 0.015 | +0.020 |
| 4.2 | Random Forest | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | `n_estimators=500`, `max_depth=8`, `min_samples_leaf=1`, `max_features=None` | **0.845** | **0.020** | **+0.050** |
| 5.1 | CatBoost | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing`; категории обработаны нативно | Параметры модели по умолчанию | 0.835 | 0.022 | +0.040 |
| 5.2 | CatBoost | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing`; категории обработаны нативно | `iterations=400`, `learning_rate=0.03`, `depth=3`, `l2_leaf_reg=0.5` | **0.842** | **0.011** | **+0.047** |
| 6.1 | FCNN | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing`; категории обработаны через `OneHotEncoder` | `hidden_size=32`, `dropout=0.2`, `Adam`, `lr=0.001`, `weight_decay=0.0001`, `epochs=100` | 0.827 | 0.027 | +0.032 |
| 7.1 | Hard Voting | Исходные признаки + `Title`, `FamilyGroup`, `Deck`, `IsChild`, `AgeMissing` | Logistic Regression + SVC + Random Forest, равные голоса | 0.839 | 0.009 | +0.044 |