#!/bin/bash
eval $(ssh-agent)
ssh-add

ECR_REPO="nvidia-picasso"
GITHASH=$(git rev-parse HEAD)
REPO_AND_TAG="${ECR_REPO}:${GITHASH}"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 426780362668.dkr.ecr.us-east-1.amazonaws.com
DOCKER_BUILDKIT=1 docker build --ssh default --no-cache -t 426780362668.dkr.ecr.us-east-1.amazonaws.com/$REPO_AND_TAG .
docker push 426780362668.dkr.ecr.us-east-1.amazonaws.com/$REPO_AND_TAG

echo "${REPO_AND_TAG}"
