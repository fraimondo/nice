.PHONY: clean

MODULES=ompk.so
PYTHON?=/usr/bin/env python
SOURCES=komplexity.c komplexity.h types.h helpers.h Makefile

all: $(MODULES)

%.so: %.c $(SOURCES)
	$(PYTHON) setup.py build
	cp build/lib.*/*.so $@
	cp $@ ../


clean:
	rm -rf build
	rm -f $(MODULES)
