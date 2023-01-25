FROM python:3.9-slim

COPY ./requirements.txt cmhg-python/requirements.txt
COPY ./app cmhg-python/app
COPY ./models cmhg-python/models
COPY ./notebooks cmhg-python/notebooks

WORKDIR /cmhg-python

RUN pip install -r requirements.txt

EXPOSE 5001

CMD ["python", "app/main.py"]