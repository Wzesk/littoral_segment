# Use matching system and python version to the Google Colab
ARG PLATFORM=linux/amd64
FROM --platform=${PLATFORM} python:3.10.12

# Set a working directory inside the container
WORKDIR /app

# Install cv2 dependences
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the requirements file first (to leverage Docker's cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Specify the default command to run your app
CMD ["python", "test_segmentation.py"]