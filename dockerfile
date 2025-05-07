FROM python:slim
RUN apt-get update && apt-get -y upgrade \
&& apt-get install -y --no-install-recommends \
git \
wget \
&& rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda PATH=/opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda update -n base -c defaults conda --yes

SHELL ["/bin/bash", "-lc"]

RUN conda create -n AML python=3.10 pip --yes -c conda-forge && \
    conda activate AML

WORKDIR /AML
COPY requirements.txt .
COPY main.py .
COPY scripts/s ./scripts/
COPY tests/ ./tests/

RUN pip install -r requirements.txt

ENV PATH=/opt/conda/envs/AML/bin:$PATH


CMD ["python", "main.py"]