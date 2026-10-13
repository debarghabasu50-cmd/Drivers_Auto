# One image for every pipeline step (AML environment is registered from it; tag = git SHA).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ARG IMAGE_TAG=unknown
ENV IMAGE_TAG=${IMAGE_TAG} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
