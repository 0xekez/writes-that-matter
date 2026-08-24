.PHONY: test verify-reference checksums paper clean

PYTHON ?= python3

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -v

verify-reference:
	PYTHONPATH=. $(PYTHON) scripts/verify_headline.py
	PYTHONPATH=. $(PYTHON) scripts/check_headline_tex.py

checksums:
	PYTHONPATH=. $(PYTHON) scripts/update_headline_checksums.py

paper: verify-reference
	cd tex && latexmk -pdf main.tex

clean:
	cd tex && latexmk -C main.tex
