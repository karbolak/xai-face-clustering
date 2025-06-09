FROM python:3.10-slim

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential \
        g++ \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh && \
    bash miniforge.sh -b -p $CONDA_DIR && \
    rm miniforge.sh

SHELL ["/bin/bash", "-c"]

RUN conda update -n base -c defaults conda --yes && \
    conda create -n AML python=3.10 pip --yes -c conda-forge && \
    conda config --add channels conda-forge

ENV PATH=/opt/conda/envs/AML/bin:$PATH

WORKDIR /AML

COPY requirements.txt .
COPY main.py .
COPY inference_api.py .
COPY streamlit_app.py .
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
