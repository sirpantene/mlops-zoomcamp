FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /srv

COPY ./requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt
