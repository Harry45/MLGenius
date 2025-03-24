FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set the working directory
WORKDIR /app

# Install any python packages you need
COPY requirements.txt ./

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the server
EXPOSE 8000
CMD ["uvicorn", "server:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
