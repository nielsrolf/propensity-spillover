# Build context = output of webapp/stage.sh (an allowlisted copy of the repo:
# inputs/, BLINDED_PROMPT.md, EXAMPLE_hypothesis.md, server/). RESULTS/,
# predictions/ and hypotheses/ are never in the context.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 BUNDLE_ROOT=/app DB_PATH=/data/spillover.db
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libyaml-dev gcc && rm -rf /var/lib/apt/lists/*
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY inputs inputs
COPY BLINDED_PROMPT.md EXAMPLE_hypothesis.md ./
COPY server server
RUN test ! -e RESULTS && test ! -e predictions && python server/build_index.py
EXPOSE 8000
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
