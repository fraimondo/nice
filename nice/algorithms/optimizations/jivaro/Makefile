.PHONY: clean

MODULES=jivaro.so
PYTHON?=/usr/bin/env python
SOURCES=blocktrie.h blocktrie.c smi.h smi.c symb_transf.h symb_transf.c helpers.h types.h Makefile

all: $(MODULES)

%.so: %.c $(SOURCES)
	$(PYTHON) setup.py build
	cp build/lib.*/*.so $@
	cp $@ ../

clean:
	rm -rf build
	rm -f $(MODULES)

MKLROOT=/opt/intel/mkl
PLATFORM = $(shell gcc -dumpmachine)
UNAME = $(shell uname)
MACHINE = $(shell uname -m)

ifeq "$(UNAME)" "Darwin"
	CC=icc
	CFLAGS = -Wl,-no_pie
	LDFLAGS= $(MKLROOT)/lib/libmkl_intel_lp64.a $(MKLROOT)/lib/libmkl_core.a $(MKLROOT)/lib/libmkl_sequential.a  -openmp-link static
endif
ifeq "$(UNAME)" "Linux"
	CFLAGS = -fPIC
	LDFLAGS= $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_sequential.a  -openmp-link static
endif

OBJS=symb_transf.o blocktrie.o smi.o
INCLUDE=-I./ -I $(MKLROOT)/include -I ../
CFLAGS +=-openmp -Wall -pedantic -O3 -m64 $(INCLUDE) -std=c99

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

test: test_perm.c $(OBJS)
	$(CC) $(INCLUDE) $(LDFLAGS) $(CFLAGS) $^ -o $@
