# File name: Dockerfile
FROM rayproject/ray:2.9.0



# Set the working dir for the container to /serve_app
WORKDIR /serve_app

# Copies the local `fake.py` file into the WORKDIR
COPY . .

RUN pip install -r requirements.txt