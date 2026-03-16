FROM qdrant/qdrant:v1.7.0

# Install wget for health checks
USER root
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

