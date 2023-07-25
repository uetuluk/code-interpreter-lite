FROM python:3.10.12 as base

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache \ 
    pip install -r /tmp/requirements.txt

COPY . /app

FROM base as default
EXPOSE 7860

CMD ["python", "app.py"]