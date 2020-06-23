# Download base image ubuntu 16.04
FROM ubuntu:latest

COPY . /

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

#RUN tar -xvf despliegue.tar
RUN pip3 install -r requeriments.txt
RUN pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
ENTRYPOINT ["python3"]
CMD ["101_prediccion_api.py"]
