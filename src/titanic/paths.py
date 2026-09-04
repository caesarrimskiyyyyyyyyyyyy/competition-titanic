from pathlib import Path


# Корень проекта относительно текущего файла.
PROJECT_DIR = Path(__file__).resolve().parents[2]


# Директория с исходными датасетами.
RAW_DATA_DIR = PROJECT_DIR/'data'/'raw'


# Директория с yaml конфигурациями моделей.
CONFIG_DIR = PROJECT_DIR/'configs'


# Директория с сохраненными ообученными моделями.
MODELS_DIR = PROJECT_DIR/'models'


# Директория с Kaggle submissions.
SUBMISSIONS_DIR = PROJECT_DIR/'submissions'


# Конфигурация итоговой модели.
FINAL_CONFIG_PATH = CONFIG_DIR/'random_forest'/'01_stable.yaml'


# Файл с обученным финальным pipeline.
FINAL_MODEL_PATH = MODELS_DIR/'random_forest.joblib'


# Итоговый файл для отправки на Kaggle.
FINAL_SUBMISSION_PATH = SUBMISSIONS_DIR/'random_forest_v1.csv'