FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
COPY model/seq2seqmodel model/seq2seqmodel


ENV PYTHONUNBUFFERED=1

RUN echo "**** install pip + requirements ****" && \
    pip install -r requirements.txt

COPY . .

CMD ["python3", "src/api.py", "--host=0.0.0.0"]
