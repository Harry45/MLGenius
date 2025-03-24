FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
RUN pip install requests
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install dill
RUN pip install fastapi
RUN pip install "uvicorn[standard]"
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "server:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]