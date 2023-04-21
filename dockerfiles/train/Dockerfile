FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest

COPY ./pipeline/
COPY requirements.txt

RUN pip3 install -r ./requirements.txt