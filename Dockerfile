FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER h2oai <ops@h2o.ai>

# Nimbix base OS
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update && \
    apt-get -y install curl && \
    curl -H 'Cache-Control: no-cache' \
    https://raw.githubusercontent.com/nimbix/image-common/master/install-nimbix.sh | bash

# Expose port 22 for local JARVICE emulation in docker
EXPOSE 22

# Notebook Common
ADD https://raw.githubusercontent.com/nimbix/notebook-common/master/install-ubuntu.sh /tmp/install-ubuntu.sh
RUN \
  bash /tmp/install-ubuntu.sh 3 && \
  rm -f /tmp/install-ubuntu.sh

# General Packaging
RUN \
  apt-get -y install \
  python-software-properties \
  software-properties-common \
  iputils-ping \
  cpio 

# Setup Repos
RUN \
  echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list && \
  gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9 && \
  gpg -a --export E084DAB9 | apt-key add -&& \
  curl -sL https://deb.nodesource.com/setup_7.x | bash - && \
  add-apt-repository ppa:fkrull/deadsnakes  && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update -yqq && \
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections


RUN \
  apt-get -y install \  
    r-base \
    r-base-dev

# Install apt-get dependancies and repos
RUN \
  apt-get install --no-install-recommends -y \
    subversion \
    gdebi-core \
    apt-utils \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler 

RUN \
  apt-get install -y \
    python-dev \
    python-pip \
    python-numpy \
    python-sklearn \
    python-skimage \
    python-scipy \
    python-setuptools \
    python-matplotlib \
    python3 \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-setuptools \
    python3-sklearn \
    python3-skimage \
    python3-matplotlib \
    python3-scipy \

# Install Oracle 8 JDK  
RUN \  
  apt-get install -y \
    oracle-java8-installer && \
  apt-get clean && \
  rm -rf /var/cache/apt/*

# Get latest h2o and gpu packages
RUN \
  mkdir /opt/caffe && \
  cd /opt && \
  pip2 install --upgrade pip && \
  pip3 install --upgrade pip && \
  pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl && \
  pip3 install \
    python-dateutil \
    pycuda \
    graphviz \
    urllib3 \
    keras && \
  pip2 install \
    pycuda \
    graphviz \
    urllib3 \
    keras

# Setup ENV
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV CAFFE_ROOT=/opt/caffe
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:/usr/local/cuda/bin:$PATH

RUN \
  cd /opt && \
  git clone https://github.com/h2oai/steam.git

# install Caffe
WORKDIR $CAFFE_ROOT
RUN git clone -b 1.0 --depth 1 https://github.com/BVLC/caffe.git . && \
    cd python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j install && \
    cd .. && \
    rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.5 \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
        -DBoost_PYTHON_LIBRARY_DEBUG=/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
        -DBoost_PYTHON_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
        -Wno-deprecated-gpu-targets \
        cd .. && \
        make -j"$(nproc)" && \
  echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig &&\
  python3 -c 'import caffe'

# caffe files alread in /opt/caffe
COPY caffe-files/caffe/ /opt/caffe-h2o

# Pip Installs
RUN \
  pip3 uninstall -y python-dateutil && \
  pip3 install \
    python-dateutil \
    pycuda \
    graphviz \
    urllib3 \
    keras && \
  pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl

# get latest files
RUN \
  mkdir -p /opt/dist && \
  cd /opt/dist && \
# download from s3
  wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/deepwater-master.tgz && \
  tar xfz deepwater-master.tgz && \
  rm deepwater-master.tgz && \
  wget -q http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/deepwater-h2o.tgz && \
  tar xfz deepwater-h2o.tgz && \
  rm deepwater-h2o.tgz && \
# install
  pip2 install tensorflow-*.whl && \
  easy_install mxnet-*.egg && \
  pip2 install h2o-*.whl && \
  pip3 install h2o-*.whl && \
  R CMD INSTALL h2o_*.tar.gz && \
  cd /opt && \
  ln -s dist/h2o.jar .

# Copy start script
COPY scripts/start-deepwater.sh /opt/start-deepwater.sh
RUN chmod +x /opt/start-deepwater.sh

# Nimbix Integrations
ADD NAE/AppDef.json /etc/NAE/AppDef.json
ADD NAE/AppDef.png /etc/NAE/default.png
ADD NAE/screenshot.png /etc/NAE/screenshot.png

WORKDIR /
EXPOSE 54321
EXPOSE 8888
EXPOSE 55000
EXPOSE 55001
EXPOSE 55002

