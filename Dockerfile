# Dockerfile

# Use a pre-built PyTorch/CUDA image to ensure all dependencies are compatible.
# This base image comes with Python, CUDA, and cuDNN pre-installed.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies. ffmpeg is crucial for video processing.
# The `yes` command automatically answers 'y' to prompts during the install.
RUN apt-get update && yes | apt-get install -y ffmpeg

# Copy the requirements file and install Python packages.
# This step is cached, so it's fast if dependencies haven't changed.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container.
COPY . .

# Set a default command to be executed when the container starts.
# It runs the main script, but the user must provide a path.
# This can be overridden by the command line.
CMD ["python", "main.py"]