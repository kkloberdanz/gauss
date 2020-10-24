CC=cc
OPT=-O3
CFLAGS=-Wall -Wextra -Wpedantic -std=gnu89 $(OPT)
SRC=$(wildcard src/*.c)
OBJ=$(patsubst src/%.c,obj/%.o,$(SRC))
INC=$(wildcard include/*.h)

libgauss.so: $(OBJ)
	$(CC) -shared -o libgauss.so $(OBJ)

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
