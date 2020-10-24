#include <stdio.h>

#include "../include/gauss.h"

int main() {
    int x = foo(42);
    if (x == 43) {
        puts("PASS");
    } else {
        puts("FAIL");
    }
}
