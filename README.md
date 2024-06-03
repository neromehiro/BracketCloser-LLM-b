# BracketCloser-LLM

最初にやること
docker pull sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1


export PROJECT_DIR=$(pwd)

docker run -it \
  -v "${PROJECT_DIR}:/app/project" \
  --workdir /app/project \
  --tmpfs /tmp:rw,size=10g \
  sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1 \
  bash

