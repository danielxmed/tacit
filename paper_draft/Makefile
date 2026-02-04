# Makefile for TACIT paper

MAIN = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex  # Run twice for references

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).toc $(MAIN).bbl $(MAIN).blg

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "PDF created: $(MAIN).pdf"
