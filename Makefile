.PHONY: install test run

install:
	pip install -r requirements.txt

test:
	python tests/test_suite.py

run:
	python main.py ask "What is the system architecture?"
