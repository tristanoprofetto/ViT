FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest

COPY ./pipeline/training ./
COPY requirements.txt ./requirements.txt

RUN pip3 install -r ./requirements.txt