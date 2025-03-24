FROM python:3.12
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# this works nicely
# docker build -t myfastapi .
# docker run -d --name mycontainer -p 8000:8000 myfastapi