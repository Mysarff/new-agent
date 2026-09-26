FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    SMARTVOYAGE_CONFIG=config/travel.json \
    SMARTVOYAGE_A2A=0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home appuser && mkdir -p /app/var && chown -R appuser:appuser /app
COPY --chown=appuser:appuser SmartVoyage ./SmartVoyage
COPY --chown=appuser:appuser config/travel.json ./config/travel.json
COPY --chown=appuser:appuser travel_knowledge ./travel_knowledge
COPY --chown=appuser:appuser app.py ./app.py
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
