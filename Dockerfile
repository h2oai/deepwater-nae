FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
MAINTAINER h2oai <ops@h2o.ai>

# base OS
ENV DEBIAN_FRONTEND noninteractive
ADD https://github.com/nimbix/image-common/archive/master.zip /tmp/nimbix.zip
WORKDIR /tmp
RUN apt-get update && apt-get -y install sudo zip unzip && unzip nimbix.zip && rm -f nimbix.zip
RUN /tmp/image-common-master/setup-nimbix.sh
RUN touch /etc/init.d/systemd-logind && apt-get -y install module-init-tools xz-utils vim openssh-server libpam-systemd libmlx4-1 libmlx5-1 iptables infiniband-diags build-essential curl libibverbs-dev libibverbs1 librdmacm1 librdmacm-dev rdmacm-utils libibmad-dev libibmad5 byacc flex git cmake screen grep && apt-get clean && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8



RUN \
  echo 'DPkg::Post-Invoke {"/bin/rm -f /var/cache/apt/archives/*.deb || true";};' | tee /etc/apt/apt.conf.d/no-cache && \
  echo "deb http://ap-northeast-1.ec2.archive.ubuntu.com/ubuntu trusty main universe" >> /etc/apt/sources.list && \
  apt-get update -q -y && \
  apt-get dist-upgrade -y && \
  apt-get clean && \
  rm -rf /var/cache/apt/* && \
# Install Oracle Java 8
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget unzip python-pip python-virtualenv python-sklearn python-pandas python-numpy python-matplotlib software-properties-common python-software-properties \
  htop libatlas3-base python-setuptools python3 python3-virtualenv r-base r-base-dev r-cran-rcurl r-cran-jsonlite curl cuda-samples-8.0 \
  build-essential cmake git libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libprotobuf-dev libsnappy-dev protobuf-compiler python3-dev python3-numpy python3-pip python3-setuptools python3-scipy  && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update -q && \
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y oracle-java8-installer && \
  apt-get clean


ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64"
ENV CAFFE_ROOT=/opt/caffe

# get latest files
RUN \
cd /opt && \
wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/h2o.jar && \
wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/deepwater-all.jar && \
wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/mxnet-0.7.0-py2.7.egg && \
wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/tensorflow-1.0.1-py2.7-linux-x86_64.egg && \
wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/h2o_3.11.0.156.tar.gz && \
wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/h2o-3.11.0.156-py2.py3-none-any.whl && \

# install python2
easy_install mxnet-0.7.0-py2.7.egg && \
easy_install tensorflow-1.0.1-py2.7-linux-x86_64.egg && \
pip install /opt/h2o-3.11.0.156-py2.py3-none-any.whl && \
R CMD INSTALL /opt/h2o_3.11.0.156.tar.gz && \
cd ..

# install Caffe
WORKDIR $CAFFE_ROOT
RUN git clone -b 1.0 --depth 1 https://github.com/BVLC/caffe.git . && \
    pip3 install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.5 \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
        -DBoost_PYTHON_LIBRARY_DEBUG=/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
        -DBoost_PYTHON_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
        .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# H2O
RUN pip3 uninstall -y python-dateutil && \
    pip3 install python-dateutil
RUN apt-get update && apt-get install -y \
        python3-skimage
RUN pip3 install --upgrade pip && pip3 install \
        pycuda

# To build the font cache
RUN python3 -c 'import caffe'
# caffe files alread in /opt/caffe
COPY caffe-files/caffe/ /opt/caffe-h2o

COPY scripts/start.sh start.sh

EXPOSE 54321

# Nimbix JARVICE emulation
EXPOSE 22
RUN mkdir -p /usr/lib/JARVICE && cp -a /tmp/image-common-master/tools /usr/lib/JARVICE
RUN cp -a /tmp/image-common-master/etc /etc/JARVICE && chmod 755 /etc/JARVICE && rm -rf /tmp/image-common-master
RUN mkdir -m 0755 /data && chown nimbix:nimbix /data
RUN sed -ie 's/start on.*/start on filesystem/' /etc/init/ssh.conf
