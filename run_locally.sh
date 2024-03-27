#!/bin/bash
eval $(ssh-agent)
ssh-add 
DOCKER_BUILDKIT=1 docker build -f Dockerfile.local --ssh default -t nvidia-picasso:dev .
docker run --network="host" -it --rm --env-file dev.env -p 8004:8004 -v ~/.cache:/root/.cache/ -v ~/.aws:/root/.aws nvidia-picasso:dev
