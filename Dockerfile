FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

RUN apt-get update && yes | apt-get install -y ffmpeg

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]