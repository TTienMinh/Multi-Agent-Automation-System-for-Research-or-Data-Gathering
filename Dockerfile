FROM python:3.11
LABEL authors="Your Name"

WORKDIR /src

COPY server.py /src/server.py
COPY arxiv_abstracts.txt /src/arxiv_abstracts.txt
COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]