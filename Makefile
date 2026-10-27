.PHONY: help clean-pyc clean-build clean test build

help:
	@echo "test        -- run the test suite"
	@echo "build       -- build sdist and wheel into dist/"
	@echo "clean       -- remove build artifacts and auxiliary python files"

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-build:
	rm -rf build dist *.egg-info

clean: clean-pyc clean-build

test:
	pytest -q

build: clean-build
	python -m build
