#!/bin/bash
# Set up the SSH agent
eval $(ssh-agent)
ssh-add 
# Build the test container
DOCKER_BUILDKIT=1 docker build -f Dockerfile.test --ssh default -t api-tests:dev .
# Set up docker to run sibling containers. We need to set a volume for the docker socket and
# also set a volume for the temporary directory used to create docker volumes internally.
sudo chmod 777 /var/run/docker.sock
docker run --network="host" --user postgres -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp --env-file test.env api-tests:dev
# # Add a volume for testing image/video generation output
