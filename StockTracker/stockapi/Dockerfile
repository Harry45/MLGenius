FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
RUN pip install requests
RUN pip install fastapi
RUN pip install "uvicorn[standard]"
RUN pip install python-dotenv
COPY . /app
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]