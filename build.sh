set -e

docker build . -t gauss-builder
docker run --rm -it gauss-builder scl enable devtoolset-8 -- bash
