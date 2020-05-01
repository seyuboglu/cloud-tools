FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
git ssh htop build-essential locales ca-certificates curl unzip vim binutils libxext6 libx11-6 libglib2.0-0 \
libxrender1 libxtst6 libxi6 tmux screen nano wget

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
RUN conda --version

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

RUN pip install --upgrade pip && pip install --upgrade gym pyyaml tqdm jupyter matplotlib wandb python-dateutil ujson \
Pillow sklearn pandas natsort seaborn grpcio tensorflow-gpu tensorflow-hub scikit-image scipy && \
pip install git+https://github.com/qubvel/classification_models.git && \
pip install git+https://github.com/tensorflow/examples.git 

CMD bash