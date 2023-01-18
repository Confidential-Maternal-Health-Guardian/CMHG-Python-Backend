**Run Server:** <br>
python app/main.py


**Docker:** <br>
docker build -t cmhg-python-backend:0.1 .
docker run -p 5000:5000 --name cmhg-python cmhg-python-backend:0.1