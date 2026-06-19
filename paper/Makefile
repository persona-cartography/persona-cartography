MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

# Figures are real build inputs: regenerating any figure must trigger a rebuild.
# (Without this, `make` only watches .tex/.bib and silently embeds stale figures.)
FIGURES := $(shell find figures -type f \( -name '*.pdf' -o -name '*.png' \) 2>/dev/null)

.PHONY: all quick clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex appendices/*.tex references.bib $(FIGURES)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

quick:
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,fls,fdb_latexmk,synctex.gz}
	rm -f sections/*.aux appendices/*.aux
