FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Commande de lancement
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
