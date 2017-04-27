OWNER:=h2o
NAME:=deepwater
BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
REV:=$(shell git rev-parse --short=10 HEAD)

image: Dockerfile
	docker build -t $(OWNER)-$(NAME):$(BRANCH) .

tag: image
	docker tag $(OWNER)-$(NAME):$(BRANCH) opsh2oai/$(OWNER)-$(NAME):$(REV)

push : tag
	docker push opsh2oai/$(OWNER)-$(NAME):$(BRANCH) && docker push opsh2oai/$(OWNER)-$(NAME):$(REV)
