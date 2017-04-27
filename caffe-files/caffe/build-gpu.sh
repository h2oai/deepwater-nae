tar --exclude cpu --exclude gpu -cf gpu/caffe.tar .
docker build --no-cache -t h2oai/deepwater:gpu gpu
