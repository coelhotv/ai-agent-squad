# Use an official Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The container will be started by docker-compose
# The 'command: tail -f /dev/null' will keep it running