FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
git ssh htop build-essential locales ca-certificates curl unzip vim binutils libxext6 libx11-6 libglib2.0-0 \
libxrender1 libxtst6 libxi6 tmux screen nano wget less zsh

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

CMD bash