FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY auth_app.py .
EXPOSE 8082
CMD ["python", "auth_app.py"]
