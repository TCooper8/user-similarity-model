FROM python:3

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY src src

ENV PORT=80
EXPOSE $PORT
CMD python3 src/main.py