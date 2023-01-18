FROM python:3.9-slim

COPY ./requirements.txt chmg-python/requirements.txt
COPY ./app chmg-python/app
COPY ./models chmg-python/models
COPY ./notebooks chmg-python/notebooks

WORKDIR /chmg-python

RUN pip install -r requirements.txt

EXPOSE 5001

CMD ["python", "app/main.py"]