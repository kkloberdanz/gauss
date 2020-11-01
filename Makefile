CC=cc
CFLAGS=-Werror -Wall -Wextra -std=gnu89 $(OPT) -fPIC
LFLAGS=-ldl -lm -lOpenCL -lclBLAS
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c,obj/%.o,$(SRC))
INC=$(wildcard src/*.h)
EXTERN_INC=$(wildcard include/*.h)

.PHONY: release
release: libgauss.so
release: OPT:=-Ofast -march=haswell -mtune=haswell -flto

.PHONY: native
native: libgauss.so
native: OPT:=-Ofast -march=native -mtune=native -flto

.PHONY: debug
debug: libgauss.so
debug: OPT:=-O0 -ggdb3

libgauss.so: $(OBJ)
	$(CC) -shared -o libgauss.so $(CFLAGS) $(OBJ) $(LFLAGS)

obj/%.o: src/%.c $(INC) $(EXTERN_INC)
	$(CC) -c -o $@ $(CFLAGS) $<

.PHONY:test
test: test/main_test

test/main_test: debug test/test.c
	$(CC) -o test/main_test test/test.c $(CFLAGS) -l:libgauss.so -L.

.PHONY: clean
clean:
	rm -f obj/*
	rm -f libgauss.so
	rm -f test/main_test
	rm -f core
