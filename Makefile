NAME:=h2o3_deepwater_nae
BRANCH:=$(shell git rev-parse --abbrev-ref HEAD)
REV:=$(shell git rev-parse --short=10 HEAD)

image: Dockerfile
	docker build -t $(NAME) .

tag: image
	docker tag $(NAME) opsh2oai/$(NAME)

push : tag
	docker push opsh2oai/$(NAME)
