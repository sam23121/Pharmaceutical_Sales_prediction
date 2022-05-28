FROM python:3.8-slim-buster 
WORKDIR /scripts 
COPY . /scripts/train.py 
RUN pip install -r ../requirements.txt 
EXPOSE 5000 
CMD ["python3","train.py"]