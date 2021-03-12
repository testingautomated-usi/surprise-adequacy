# =======================================
# This is an automatically generated file.
# =======================================
FROM tensorflow/tensorflow:nightly-gpu

# Update pip and install all pip dependencies
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install dataclasses
COPY requirements.txt /opt/project/requirements.txt
COPY test_requirements.txt /opt/project/test_requirements.txt
COPY study_requirements.txt /opt/project/study_requirements.txt
RUN pip install -r /opt/project/requirements.txt
RUN pip install -r /opt/project/test_requirements.txt
RUN pip install -r /opt/project/study_requirements.txt

# Required for foolbox
RUN apt-get -y install git