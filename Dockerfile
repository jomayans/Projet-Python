FROM ubuntu:22.04
WORKDIR ${HOME}/cine_insights
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install mlflow
COPY main.py .
COPY src ./src
CMD ["python3", "script.py"]