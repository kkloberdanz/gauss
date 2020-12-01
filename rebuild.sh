make clean
ls src/* | entr sh -c "make $1 -j && cp libgauss.so ../pygauss/gauss/lib/. && echo DONE"
