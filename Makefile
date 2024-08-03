SHELL = bash

# Install pre-commit hooks
.PHONY: install-hooks
install-hooks:
	pre-commit install

# Run pre-commit hooks
.PHONY: pre-commit-hooks
pre-commit-hooks:
	pre-commit run --all-files

# Styling
.PHONY: style
style:
	black . --exclude=venv/
	python -m isort . --skip=venv
	pydocstringformatter . --write --style=pep257 --style=numpydoc --exclude=venv/**/*
	flake8 . --exclude=venv/*,"/a_pp__.py" --ignore=E501,W503,E226,E203
	pyupgrade

# Combined target to run both styling and pre-commit hooks
.PHONY: check
check: pre-commit-hooks style

# Cleaning
.PHONY: clean
clean: check
	find . -type f -name "*.DS_Store" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
	rm -rf .coverage*
