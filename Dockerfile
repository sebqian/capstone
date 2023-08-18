FROM pytorch/pytorch:latest
COPY requirements.txt /tmp/requirements.txt

# Install other packages
RUN apt-get update -y && \
	apt-get install -y gcc && \
	pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    mkdir -p /workspace/codebase && \
    mkdir /workspace/data
WORKDIR /workspace/codebase

# Activate the Conda environment
ENV PYTHONPATH=/workspace:$PYTHONPATH
