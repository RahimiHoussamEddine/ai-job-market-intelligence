FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JOB_MARKET_PROJECT_ROOT=/opt/app \
    JOB_INTELLIGENCE_ARTIFACT_DIR=/opt/artifacts/v4 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /opt/app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY requirements.lock pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
COPY app ./app
COPY configs ./configs

RUN python -m pip install --no-cache-dir --requirement requirements.lock \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && mkdir -p /opt/artifacts /home/app/.streamlit \
    && chown -R app:app /opt/artifacts /home/app/.streamlit

USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
