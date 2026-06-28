FROM python:3.11-slim

WORKDIR /app

# Build tools + OpenJDK for Apache Tika
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME so tika-python can find the JVM
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir pipenv

COPY Pipfile ./
RUN pipenv install --system --deploy --ignore-pipfile

# Pre-download the Tika JAR so first-request latency is zero
RUN python -c "import tika; tika.initVM()" 2>/dev/null || true

COPY patternengine/ ./patternengine/
COPY hudex_demo.html ./

WORKDIR /app/patternengine

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
