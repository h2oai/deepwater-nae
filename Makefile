NAME:=h2o3_deepwater_nae
BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
REV:=$(shell git rev-parse --short=10 HEAD)

image: Dockerfile
	docker build -t $(NAME):$(BRANCH) .

tag: image
	docker tag $(NAME):$(BRANCH) opsh2oai/$(NAME):$(REV)

push : tag
	docker push opsh2oai/$(NAME):$(BRANCH) && docker push opsh2oai/$(NAME):$(REV)
