FROM python:3.10-slim

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# Miniconda install
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

SHELL ["/bin/bash", "-c"]

RUN conda update -n base -c defaults conda --yes && \
    conda create -n AML python=3.10 pip --yes -c conda-forge

ENV PATH /opt/conda/envs/AML/bin:$PATH

WORKDIR /AML

COPY requirements.txt .
COPY main.py .
COPY inference_api.py .
COPY streamlit_app.py .
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -r requirements.txt

# Default command - overridden at runtime
CMD ["bash"]
