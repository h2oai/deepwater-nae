FROM bvlc/caffe:gpu

RUN apt-get update && apt-get install -y \
        python-skimage

RUN pip install --upgrade pip && pip install \
        pycuda

# To build the font cache
RUN python -c 'import caffe'

RUN mkdir -p /h2o-docker/caffe
ADD caffe.tar /h2o-docker/caffe/
