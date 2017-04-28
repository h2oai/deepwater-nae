FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
MAINTAINER h2oai <ops@h2o.ai>

# Nimbix base OS
ENV DEBIAN_FRONTEND noninteractive
ADD https://github.com/nimbix/image-common/archive/master.zip /tmp/nimbix.zip
WORKDIR /tmp
RUN apt-get update && apt-get -y install sudo zip unzip && unzip nimbix.zip && rm -f nimbix.zip
RUN /tmp/image-common-master/setup-nimbix.sh
RUN touch /etc/init.d/systemd-logind && apt-get -y install \
  locales \
  module-init-tools \
  xz-utils \
  vim \
  openssh-server \
  libpam-systemd \
  libmlx4-1 \
  libmlx5-1 \
  iptables \
  infiniband-diags \
  build-essential \
  curl \
  libibverbs-dev \
  libibverbs1 \
  librdmacm1 \
  librdmacm-dev \
  rdmacm-utils \
  libibmad-dev \
  libibmad5 \
  byacc \
  flex \
  git \
  cmake \
  screen \
  apt-utils \
  software-properties-common \
  wget \
  grep

# Clean up image
RUN apt-get clean
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Install apt-get dependancies and repos
RUN \
  echo 'DPkg::Post-Invoke {"/bin/rm -f /var/cache/apt/archives/*.deb || true";};' | tee /etc/apt/apt.conf.d/no-cache && \
  echo "deb http://ap-northeast-1.ec2.archive.ubuntu.com/ubuntu xenial main universe" >> /etc/apt/sources.list && \
  echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update -q -y && \
  apt-get install --no-install-recommends -y \
    subversion \
    gdebi-core \
    libatlas3-base \
    r-base \
    r-base-dev \
    r-cran-rcurl \
    r-cran-jsonlite \
    cuda-samples-8.0 \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-software-properties \
    python-setuptools \
    python-numpy \
    python-scipy \
    python-matplotlib \
    python-sklearn \
    python-pip \
    python3 \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-setuptools \
    python3-sklearn \
    python3-skimage \
    python3-matplotlib \
    python3-scipy && \
  rm -rf /var/cache/apt/* && \
  apt-get clean

# Install Oracle 8 JDK  
RUN \  
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
  apt-get install -y oracle-java8-installer

# Install RStudio
RUN \
  wget https://download2.rstudio.org/rstudio-server-1.0.143-amd64.deb && \
  gdebi -n rstudio-server-1.0.143-amd64.deb && \
  rm rstudio-server-1.0.143-amd64.deb

# Get latest h2o and gpu packages
RUN \
  mkdir /opt/caffe && \
  cd /opt && \
  pip3 install --upgrade pip && \
  wget http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/h2o.jar && \ 
  wget http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/deepwater-all.jar  && \
  wget http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/mxnet-0.7.0-py2.7.egg && \
  easy_install mxnet-0.7.0-py2.7.egg 

# Setup ENV
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64"
ENV CAFFE_ROOT=/opt/caffe
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH

# install Caffe
WORKDIR /opt/caffe
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
  echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Pip Installs
RUN \
  pip3 install --upgrade pip && \
  pip3 uninstall -y python-dateutil && \
  pip3 install python-dateutil && \
  pip3 install pycuda

RUN python3 -c 'import caffe'

# caffe files alread in /opt/caffe
COPY caffe-files/caffe/ /opt/caffe-h2o

# Copy start script
COPY ./scripts/start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

# Nimbix Integrations
ADD ./NAE/AppDef.json /etc/NAE/AppDef.json
ADD ./NAE/AppDef.png /etc/NAE/default.png
ADD ./NAE/screenshot.png /etc/NAE/screenshot.png
ADD ./NAE/url.txt /etc/NAE/url.txt

WORKDIR /opt
EXPOSE 54321

# Nimbix JARVICE emulation
EXPOSE 22
RUN mkdir -p /usr/lib/JARVICE && cp -a /tmp/image-common-master/tools /usr/lib/JARVICE
RUN cp -a /tmp/image-common-master/etc /etc/JARVICE && chmod 755 /etc/JARVICE && rm -rf /tmp/image-common-master
RUN mkdir -m 0755 /data && chown nimbix:nimbix /data
RUN sed -ie 's/start on.*/start on filesystem/' /etc/init/ssh.conf

