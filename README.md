# nps_significance_testing
This repo contains the accompanying code to the Medium article [xxx](xxx), detailing how you can use permutation 
testing to determine whether the [NPS](https://en.wikipedia.org/wiki/Net_Promoter) scores collected from two groups 
are significantly different.

## Pre-requisites

### Option 1 (Local machine)

#### Python

You will need Python 3 available on your machine, which can be installed [here](https://www.python.org/downloads/).

#### Virtual Environment

It is recommended to install the required libraries within a virtual environment as per the commands below:

```shell script
# Create virtual environment
python3 -m venv .venv

# Enter virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2 (Docker)

The code can also be executed and tested using the docker image provided. You will need to have 
[Docker](https://docs.docker.com/get-docker/) installed on your machine, and to make sure this directory can be bind 
mounted into Docker containers. 

```shell script
# Build docker image 
docker-compose build

# Set up container
docker-compose up -d

# Run unit tests
docker exec nps_sig_dev /home/docker_user/.local/bin/pytest --verbose tests

# Enter container
docker exec -it nps_sig_dev bash
```

## Running a significance test

The notebook [Example_Notebook](Example_Notebook.ipynb) gives an overview of how the code can be used, which makes use 
of the source code in [significance_testing.py](significance_testing.py)
