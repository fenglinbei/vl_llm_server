FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
COPY requirements.txt /workspace/
COPY . /workspace

RUN pip install --no-cache-dir -r /workspace/requirements.txt -i https://mirror.baidu.com/pypi/simple

EXPOSE 5006