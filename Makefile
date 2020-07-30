.PHONY: lint
lint:
	pipenv run flake8 *.py

.PHONY: typecheck
typecheck:
	pipenv run mypy *.py

.PHONY: unit
unit: 
	pipenv run python -m pytest -v

.PHONY: test
test: lint typecheck unit

.PHONY: isort
isort:
	pipenv run isort *.py groa_ds_api/*.py

.PHONY: pipfile
pipfile:
	pipenv install

.PHONY: requirements
requirements:
	pipenv run pip freeze > requirements.txt