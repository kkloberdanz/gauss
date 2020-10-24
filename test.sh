set -e

make test -j$(nproc)
LD_LIBRARY_PATH=`pwd` ./test/main_test
