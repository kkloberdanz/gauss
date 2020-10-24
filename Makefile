CC=cc
OPT=-O3
CFLAGS=-Wall -Wextra -Wpedantic -std=gnu89 $(OPT) -fPIC
LFLAGS=
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c,obj/%.o,$(SRC))
INC=$(wildcard include/*.h)

.PHONY: release
release: libgauss.so
release: OPT:=-O3 -flto

.PHONY: debug
debug: libgauss.so
debug: OPT:=-O0 -ggdb3

libgauss.so: $(OBJ)
	$(CC) -shared -o libgauss.so $(CFLAGS) $(OBJ) $(LFLAGS)

$(OBJ): $(SRC) $(INC)
	$(CC) -c -o $@ $(CFLAGS) $<

.PHONY:test
test: test/main_test

test/main_test: test/test.c libgauss.so
	$(CC) -o test/main_test test/test.c -l:libgauss.so -L.

.PHONY: clean
clean:
	rm -f obj/*
	rm -f libgauss.so
	rm -f test/main_test
