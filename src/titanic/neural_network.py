from torch import nn


class TitanicFCNN(nn.Module):
    """
    Небольшая полносвязная сеть для бинарной классификации.

    Размер входного слоя определяется автоматически после
    preprocessing, поэтому сеть не зависит от количества
    колонок, созданных OneHotEncoder.
    """

    def __init__(self, hidden_size=32, dropout=0.2):
        super().__init__()

        self.layers = nn.Sequential(
            # LazyLinear определит число входных признаков
            # при первом проходе данных через сеть.
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Два выхода, что соответствует задаче 
            # бинарной классификации.
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, data):
        return self.layers(data)