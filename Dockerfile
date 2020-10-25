FROM centos:7

RUN yum install -y centos-release-scl

RUN yum install -y \
    python3-devel \
    make \
    devtoolset-8-gcc

WORKDIR /build

COPY . /build/
