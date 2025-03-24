FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]