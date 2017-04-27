FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
MAINTAINER h2oai <ops@h2o.ai>

# base OS
ENV DEBIAN_FRONTEND noninteractive
ADD https://github.com/nimbix/image-common/archive/master.zip /tmp/nimbix.zip
WORKDIR /tmp
RUN apt-get update && apt-get -y install sudo zip unzip && unzip nimbix.zip && rm -f nimbix.zip
RUN /tmp/image-common-master/setup-nimbix.sh
RUN touch /etc/init.d/systemd-logind && apt-get -y install locales module-init-tools xz-utils vim openssh-server libpam-systemd libmlx4-1 libmlx5-1 iptables infiniband-diags build-essential curl libibverbs-dev libibverbs1 librdmacm1 librdmacm-dev rdmacm-utils libibmad-dev libibmad5 byacc flex git cmake screen grep
RUN apt-get clean
RUN locale-gen en_US.UTF-8 && \ 
update-locale LANG=en_US.UTF-8

RUN \
  echo 'DPkg::Post-Invoke {"/bin/rm -f /var/cache/apt/archives/*.deb || true";};' | tee /etc/apt/apt.conf.d/no-cache && \
  echo "deb http://ap-northeast-1.ec2.archive.ubuntu.com/ubuntu trusty main universe" >> /etc/apt/sources.list && \
  apt-get update -q -y && \
# Install Oracle Java 8
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget apt-utils unzip python-pip python-virtualenv python-sklearn python-pandas python-numpy python-matplotlib software-properties-common python-software-properties \
  htop libatlas3-base python-setuptools python3 python3-virtualenv r-base r-base-dev r-cran-rcurl r-cran-jsonlite curl cuda-samples-8.0 \
  build-essential cmake git libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libprotobuf-dev libsnappy-dev protobuf-compiler python3-dev python3-numpy python3-pip python3-setuptools python3-scipy  && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update -q && \
  echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y oracle-java8-installer && \
  rm -rf /var/cache/apt/* && \
  apt-get clean

RUN \
  echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list && \
  gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9 && \
  gpg -a --export E084DAB9 | apt-key add -&& \
  apt-get update -q -y && \
  apt-get install -y r-base r-base-dev

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64"
ENV CAFFE_ROOT=/opt/caffe

# get latest files
RUN \
  mkdir /opt/caffe && \
  cd /opt && \
  pip3 install --upgrade pip && \
  wget http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/h2o.jar && \ 
  wget http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/deepwater-all.jar  && \
  wget http://s3.amazonaws.com/h2o-deepwater/public/nightly/latest/mxnet-0.7.0-py2.7.egg

RUN \
  cd /opt && \
  easy_install mxnet-0.7.0-py2.7.egg 

# install Caffe
WORKDIR /opt/caffe
RUN git clone -b 1.0 --depth 1 https://github.com/BVLC/caffe.git . && \
    cd python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.5 \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
        -DBoost_PYTHON_LIBRARY_DEBUG=/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
        -DBoost_PYTHON_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
        cd .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# To build the font cache
RUN pip3 uninstall -y python-dateutil && \
    pip3 install python-dateutil
RUN apt-get update && apt-get install -y \
        python3-skimage
RUN pip3 install --upgrade pip && pip3 install \
        pycuda
RUN python3 -c 'import caffe'

# caffe files alread in /opt/caffe
COPY caffe-files/caffe/ /opt/caffe-h2o

# Copy start script
ADD ./scripts/start.sh /tmp/start.sh
RUN chmod +x /tmp/start.sh

# Nimbix Integrations
ADD ./NAE/AppDef.json /etc/NAE/AppDef.json
ADD ./NAE/AppDef.png /etc/NAE/default.png
ADD ./NAE/url.txt /etc/NAE/url.txt

EXPOSE 54321
# Nimbix JARVICE emulation
EXPOSE 22
RUN mkdir -p /usr/lib/JARVICE && cp -a /tmp/image-common-master/tools /usr/lib/JARVICE
RUN cp -a /tmp/image-common-master/etc /etc/JARVICE && chmod 755 /etc/JARVICE && rm -rf /tmp/image-common-master
RUN mkdir -m 0755 /data && chown nimbix:nimbix /data
RUN sed -ie 's/start on.*/start on filesystem/' /etc/init/ssh.conf

USER nimbix
CMD ["/tmp/start.sh"]
