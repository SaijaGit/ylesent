## YleSent

### Clone project
```bash
git clone https://github.com/SaijaGit/ylesent.git
cd ylesent
```

### Poetry
```bash
poetry env use 3.13
poetry install
```

### Run all
```bash
poetry run python -m src
```

### Run specific parts
```bash
poetry run python -m src preprocess
poetry run python -m src train
poetry run python -m src predict
poetry run python -m src visualize
```