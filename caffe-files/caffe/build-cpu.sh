tar --exclude cpu --exclude gpu -cf cpu/caffe.tar .
docker build --no-cache -t h2oai/deepwater:cpu cpu
