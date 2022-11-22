# engineering_practices_ml

## Установка пакетного менеджера (poetry)

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

## Развёртывание окружения:
1) prod

```shell
poetry install
```
2) dev

```shell
poetry install --without dev
```

## Сборка пакета

```shell
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi <PYPI-TOKEN>
poetry build
poetry publish -r test-pypi
```

## Ссылка на пакет в pypi-test

```
https://test.pypi.org/project/liza-knn/
```

## Установка пакета из pypi-test

```shell
pip install -i https://test.pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ liza-knn
```

## Запуск
```shell
cd src
python3 main.py
```
