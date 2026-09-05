# =============================================================================
# PRODUCTION-GRADE DOCKERFILE FOR AGENTIC STREAMLIT PRICING APP
# =============================================================================
# Stable, slim Debian-based Python image for minimal size and security compliance
FROM python:3.12-slim

# Set system-level environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/app:/app/src" \
    PORT=8501

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (including libgomp1 for XGBoost and curl for Streamlit healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install Python dependencies with pip configuration optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and directories to the container
COPY src/ ./src/
COPY docs/ ./docs/
COPY outputs/ ./outputs/
COPY pricing_functions_v2.py .
COPY app.py ./pricing_app.py
COPY app.py .

# Create a non-privileged system user to run Streamlit for enhanced security compliance
RUN useradd -u 8501 -m streamlit && \
    mkdir -p /app/data && \
    chown -R streamlit:streamlit /app

# Switch to the non-root user
USER streamlit

# Expose port 8501 for Streamlit traffic
EXPOSE 8501

# Implement a robust Docker container healthcheck using Streamlit's built-in status endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application using production-ready configuration flags
CMD ["streamlit", "run", "pricing_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]
