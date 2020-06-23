# Download base image ubuntu 16.04
FROM ubuntu:latest

COPY . /

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
	
#RUN tar -xvf despliegue.tar
RUN pip3 install -r requeriments.txt
ENTRYPOINT ["python3"]
CMD ["cesens_api.py"]
