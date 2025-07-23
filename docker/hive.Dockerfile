# Hive Orchestrator Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/hive_orchestrator/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the omega_phr library
COPY omega_phr/ /app/omega_phr/

# Copy service files
COPY services/hive_orchestrator/ /app/

# Generate protobuf files
RUN python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=. \
    --grpc_python_out=. \
    proto/hive.proto

# Create non-root user
RUN useradd --create-home --shell /bin/bash hive
RUN chown -R hive:hive /app
USER hive

# Expose the service port
EXPOSE 50052

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; from grpc_health.v1 import health_pb2_grpc, health_pb2; \
    channel = grpc.insecure_channel('localhost:50052'); \
    health_stub = health_pb2_grpc.HealthStub(channel); \
    request = health_pb2.HealthCheckRequest(service='hive'); \
    response = health_stub.Check(request)" || exit 1

# Start the service
CMD ["python", "main.py"]
