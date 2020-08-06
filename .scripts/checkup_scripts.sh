# Run autopep8 to fix code
poetry run autopep8 --recursive --aggressive --in-place .

# Check other requirements of PEP8
poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
poetry run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=79 --statistics

# Check mypy
poetry run mypy --ignore-missing-imports .
