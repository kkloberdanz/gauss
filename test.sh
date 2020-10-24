set -e

make test
LD_LIBRARY_PATH=`pwd` ./test/main_test
