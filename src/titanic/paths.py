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