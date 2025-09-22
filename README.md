Clone project

Poetry:
poetry env use 3.13
poetry install

Run all:
poetry run python -m src

Run speficic parts:
poetry run python -m src preprocess
poetry run python -m src train
poetry run python -m src predict
poetry run python -m src visualize