
tests: build 
	@echo python3 -m pytest $@

build:
	@pip install .
.PHONY: tests build
