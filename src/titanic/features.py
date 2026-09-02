# Разные варианты одинаковых обращений
# приводим к единому написанию.
_TITLE_ALIASES = {
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs'
}


# Частые обращения оставляем отдельными категориями.
# Все остальные обращения будем объединять в 'Rare'.
_COMMON_TITLES = {
    'Mr',
    'Miss',
    'Mrs',
    'Master'
}


# Перед применением словаря размер семьи ограничивается
# сверху значением 5. Ключ 5 означат семью из 5 и более.
_FAMILY_GROUP_BY_CAPPED_SIZE = {
    1: 'Alone',
    2: 'Small',
    3: 'Small',
    4: 'Small',
    5: 'Large'
}


# Эти поля используются для создания новых признаков,
# но непосредственно в модель не передаются.
_DROPPED_FEATURES = {
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin'
}


def _pclass(data):
    """
    Преобразует класс билета в категориальный признак.
    """

    return (
        data.Pclass
        .astype(str)
    )


def _title(data):
    """
    Извлекает обращение пассажира из имени.
    """

    return (
        data.Name

        # Извлекает текст между запятой и точкой.
        .str.extract(r',\s*([^.]*)\.', expand=False)

        # Удаляет возможные пробелы по краям.
        .str.strip()

        # Объединяем разные варианты одинаковых обращений.
        .replace(_TITLE_ALIASES)

        # Редкие обращения заменяем категорией Rare.
        .where(
            lambda title: title.isin(_COMMON_TITLES),
            'Rare'
        )
    )


def _family_group(data):
    """
    Создает категорию размера семьи на борту.

    Размер семьи: 
    SibSp + Parch + 1 (сам пассажир).
    """

    return (
        data.SibSp

        # Добавляем родителей и детей.
        .add(data.Parch)

        # Добавляем самого пассажира.
        .add(1)

        # Все размеры 5+ приводим к значению 5.
        .clip(upper=5)

        # Преобразуем размер в категорию.
        .map(_FAMILY_GROUP_BY_CAPPED_SIZE)
    )


def _deck(data):
    """
    Извлекает палубу из первой буквы номера каюты.

    Если каюта неизвестна, используется категория Unknown.
    """

    return (
        data.Cabin

        # Извлекает первую латинскую букву.
        .str.extract(r'^([A-Za-z])', expand=False)

        # Приводим букву к верхнему регистру.
        .str.upper()

        # Отсутствие каюты сохраняем как отдельную категорию.
        .fillna('Unknown')
    )


def _is_child(data):
    """
    Отмечает пассажиров возрастом не старше 12 лет.
    """

    return (
        data.Age
        .le(12)
        .astype('int8')
    )


def _age_missing(data):
    """
    Отмечает пассажиров с неизвестным возрастом.
    """

    return (
        data.Age
        .isna()
        .astype('int8')
    )


# Декларативно описываем создаваемые признаки.
# Ключ: название итогового столбца, значение:
# функция, вычисляющая этот столбец.
_FEATURE_BUILDERS = {
    'Pclass': _pclass,
    'Title': _title,
    'FamilyGroup': _family_group,
    'Deck': _deck,
    'IsChild': _is_child,
    'AgeMissing': _age_missing
}


def make_features(data):
    """
    Создает полный набор признаков для модели.
    """

    return (
        data

        # Вызываем функции из _FEATURE_BUILDERS
        # и добавляем рассчитанные столбцы.
        .assign(**_FEATURE_BUILDERS)

        # Удаляем идентификаторы и исходные текстовые поля
        # после извлечения полезной информации.
        .drop(columns=_DROPPED_FEATURES)
    )