FROM python:3.6.10-slim-buster
EXPOSE 5001

RUN apt-get update
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip
RUN apt-get install -y git

RUN git clone https://dpolob:"P289bt25!"@github.com/dpolob/WeatherPrediction.git

WORKDIR /WeatherPrediction
RUN pip3 install -r requeriments.txt
RUN pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
