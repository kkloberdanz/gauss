CC=cc
OPT=-O3
CFLAGS=-Werror -Wall -Wextra -std=gnu89 $(OPT) -fPIC
LFLAGS=-ldl
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
	$(CC) -c -o $@ $(CFLAGS) $< $(LFLAGS)

.PHONY:test
test: test/main_test

test/main_test: OPT:=-O0
test/main_test: test/test.c libgauss.so
	$(CC) -o test/main_test test/test.c $(CFLAGS) -l:libgauss.so -L.

.PHONY: clean
clean:
	rm -f obj/*
	rm -f libgauss.so
	rm -f test/main_test
