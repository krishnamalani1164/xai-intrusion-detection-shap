# Deployment Guide

## Overview

This guide covers deploying the XAI Intrusion Detection System in various production environments, from single-server deployments to distributed cloud architectures.

## Deployment Options

### 1. Standalone Server Deployment

#### System Requirements
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB minimum (32GB+ for large datasets)
- **Storage**: 100GB+ SSD
- **OS**: Ubuntu 20.04+, CentOS 8+, or RHEL 8+

#### Installation Steps

1. **Server Setup**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install system dependencies
   sudo apt install -y python3.9 python3.9-venv python3.9-dev \
                       build-essential libssl-dev libffi-dev \
                       nginx supervisor redis-server
   ```

2. **Application Deployment**
   ```bash
   # Create application user
   sudo useradd -m -s /bin/bash xai-ids
   sudo su - xai-ids
   
   # Clone and setup application
   git clone https://github.com/yourusername/xai-intrusion-detection.git
   cd xai-intrusion-detection
   python3.9 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

3. **Configuration**
   ```bash
   # Copy production configs
   cp configs/production/model_config.yaml configs/
   cp configs/production/data_config.yaml configs/
   
   # Set environment variables
   export XAI_IDS_ENV=production
   export XAI_IDS_DATA_PATH=/var/lib/xai-ids/data
   export XAI_IDS_MODEL_PATH=/var/lib/xai-ids/models
   ```

### 2. Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 xai-ids && chown -R xai-ids:xai-ids /app
USER xai-ids

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command
CMD ["python", "src/api_server.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  xai-ids:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - XAI_IDS_ENV=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - xai-ids
    restart: unless-stopped

volumes:
  redis_data:
```

#### Deployment Commands
```bash
# Build and start services
docker-compose up -d

# Scale the application
docker-compose up -d --scale xai-ids=3

# View logs
docker-compose logs -f xai-ids

# Update deployment
docker-compose pull && docker-compose up -d
```

### 3. Kubernetes Deployment

#### Namespace and ConfigMap
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: xai-ids

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: xai-ids-config
  namespace: xai-ids
data:
  model_config.yaml: |
    model:
      algorithm: "random_forest"
      hyperparameters:
        n_estimators: 200
        max_depth: 15
    training:
      test_size: 0.2
      validation_split: 0.1
```

#### Deployment and Service
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xai-ids-deployment
  namespace: xai-ids
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xai-ids
  template:
    metadata:
      labels:
        app: xai-ids
    spec:
      containers:
      - name: xai-ids
        image: your-registry/xai-ids:latest
        ports:
        - containerPort: 8000
        env:
        - name: XAI_IDS_ENV
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        volumeMounts:
        - name: config-volume
          mountPath: /app/configs
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: xai-ids-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: xai-ids-data-pvc

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: xai-ids-service
  namespace: xai-ids
spec:
  selector:
    app: xai-ids
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Persistent Volume
```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: xai-ids-data-pvc
  namespace: xai-ids
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### 4. Cloud Deployments

#### AWS Deployment

**Using ECS with Fargate:**
```json
{
  "family": "xai-ids-task",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "xai-ids",
      "image": "your-account.dkr.ecr.region.amazonaws.com/xai-ids:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "XAI_IDS_ENV",
          "value": "production"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-west-2"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/xai-ids",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Using Lambda for Serverless:**
```python
# lambda_handler.py
import json
import base64
import numpy as np
from src.real_time_detector import RealTimeDetector

# Initialize detector (loaded once per container)
detector = RealTimeDetector(
    model_path='/opt/models/trained_model.pkl',
    preprocessor_path='/opt/models/preprocessor.pkl'
)

def lambda_handler(event, context):
    try:
        # Parse input data
        if 'body' in event:
            data = json.loads(event['body'])
        else:
            data = event
        
        # Convert to numpy array
        X = np.array(data['features']).reshape(1, -1)
        
        # Make prediction with explanation
        prediction, probability, explanation = detector.predict_with_explanation(X)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction[0]),
                'probability': float(probability[0]),
                'explanation': explanation.tolist() if hasattr(explanation, 'tolist') else str(explanation)
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Google Cloud Deployment

**Using Cloud Run:**
```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: xai-ids
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-boost: "true"
        run.googleapis.com/memory: "8Gi"
        run.googleapis.com/cpu: "4"
    spec:
      containers:
      - image: gcr.io/your-project/xai-ids:latest
        ports:
        - containerPort: 8000
        env:
        - name: XAI_IDS_ENV
          value: production
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
```

#### Azure Deployment

**Using Container Instances:**
```yaml
# azure-container-instance.yaml
apiVersion: 2019-12-01
location: eastus
name: xai-ids-container
properties:
  containers:
  - name: xai-ids
    properties:
      image: your-registry.azurecr.io/xai-ids:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: XAI_IDS_ENV
        value: production
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
```

## Production Configuration

### 1. Environment Configuration

**Production Environment Variables:**
```bash
# Application settings
export XAI_IDS_ENV=production
export XAI_IDS_DEBUG=false
export XAI_IDS_LOG_LEVEL=INFO

# Data paths
export XAI_IDS_DATA_PATH=/var/lib/xai-ids/data
export XAI_IDS_MODEL_PATH=/var/lib/xai-ids/models
export XAI_IDS_OUTPUT_PATH=/var/lib/xai-ids/outputs

# Database settings
export DATABASE_URL=postgresql://user:pass@localhost:5432/xai_ids
export REDIS_URL=redis://localhost:6379

# Security settings
export SECRET_KEY=your-secret-key-here
export JWT_SECRET=your-jwt-secret-here
export API_KEY_HASH=your-api-key-hash

# Performance settings
export MAX_WORKERS=4
export BATCH_SIZE=1000
export CACHE_TTL=3600

# Monitoring
export ENABLE_METRICS=true
export METRICS_PORT=9090
export HEALTH_CHECK_INTERVAL=30
```

### 2. Security Configuration

**SSL/TLS Setup:**
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://xai-ids-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        access_log off;
        proxy_pass http://xai-ids-backend/health;
    }
}

upstream xai-ids-backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}
```

**API Authentication:**
```python
# src/auth.py
from functools import wraps
import jwt
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_jwt(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token required'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer '
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function
```

### 3. Database Configuration

**PostgreSQL Setup:**
```sql
-- Create database and user
CREATE DATABASE xai_ids;
CREATE USER xai_ids_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE xai_ids TO xai_ids_user;

-- Create tables
\c xai_ids;

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    features JSONB,
    prediction INTEGER,
    probability FLOAT,
    explanation JSONB,
    model_version VARCHAR(50)
);

CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    model_name VARCHAR(100),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT
);

CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);
```

### 4. Monitoring and Logging

**Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'xai-ids'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
```

**Grafana Dashboard:**
```json
{
  "dashboard": {
    "title": "XAI-IDS Monitoring",
    "panels": [
      {
        "title": "Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(xai_ids_predictions_total[5m])"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "xai_ids_model_accuracy"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, xai_ids_request_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

**Logging Configuration:**
```python
# src/logging_config.py
import logging
import logging.config
from pythonjsonlogger import jsonlogger

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': jsonlogger.JsonFormatter,
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': '/var/log/xai-ids/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## High Availability Setup

### 1. Load Balancing

**HAProxy Configuration:**
```
# haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend xai_ids_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/xai-ids.pem
    redirect scheme https if !{ ssl_fc }
    default_backend xai_ids_backend

backend xai_ids_backend
    balance roundrobin
    option httpchk GET /health
    server xai-ids-1 192.168.1.10:8000 check
    server xai-ids-2 192.168.1.11:8000 check
    server xai-ids-3 192.168.1.12:8000 check
```

### 2. Database Replication

**Master-Slave PostgreSQL:**
```bash
# Master configuration (postgresql.conf)
wal_level = replica
max_wal_senders = 3
checkpoint_segments = 8
wal_keep_segments = 8

# Slave configuration
hot_standby = on
```

### 3. Redis Clustering

**Redis Sentinel:**
```
# sentinel.conf
port 26379
sentinel monitor xai-ids-redis 192.168.1.20 6379 2
sentinel down-after-milliseconds xai-ids-redis 5000
sentinel failover-timeout xai-ids-redis 10000
sentinel parallel-syncs xai-ids-redis 1
```

## Performance Optimization

### 1. Model Optimization

**Model Serving Optimization:**
```python
# src/optimized_detector.py
import onnx
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import numpy as np

class OptimizedDetector:
    def __init__(self, model_path):
        # Convert scikit-learn model to ONNX for faster inference
        sklearn_model = joblib.load(model_path)
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            sklearn_model, 
            "RandomForestClassifier",
            [("input", FloatTensorType([None, sklearn_model.n_features_in_]))]
        )
        
        # Create ONNX runtime session
        self.session = ort.InferenceSession(onnx_model.SerializeToString())
        
    def predict(self, X):
        input_name = self.session.get_inputs()[0].name
        return self.session.run(None, {input_name: X.astype(np.float32)})[0]
```

### 2. Caching Strategy

**Redis Caching:**
```python
# src/cache.py
import redis
import pickle
import hashlib
import json

class PredictionCache:
    def __init__(self, redis_url, ttl=3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl
    
    def get_cache_key(self, features):
        # Create hash of features for cache key
        features_str = str(sorted(features.flatten()))
        return hashlib.md5(features_str.encode()).hexdigest()
    
    def get_prediction(self, features):
        key = self.get_cache_key(features)
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    def cache_prediction(self, features, prediction, probability, explanation):
        key = self.get_cache_key(features)
        data = {
            'prediction': prediction,
            'probability': probability,
            'explanation': explanation
        }
        self.redis.setex(key, self.ttl, pickle.dumps(data))
```

### 3. Batch Processing

**Async Batch Processor:**
```python
# src/batch_processor.py
import asyncio
import aioredis
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AsyncBatchProcessor:
    def __init__(self, model, batch_size=1000, max_workers=4):
        self.model = model
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results_queue = asyncio.Queue()
    
    async def process_batch(self, data_queue):
        """Process batches of data asynchronously"""
        batch = []
        batch_ids = []
        
        # Collect batch
        while len(batch) < self.batch_size:
            try:
                item = await asyncio.wait_for(data_queue.get(), timeout=1.0)
                batch.append(item['features'])
                batch_ids.append(item['id'])
            except asyncio.TimeoutError:
                if batch:
                    break
                continue
        
        if batch:
            # Process batch in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor, 
                self._process_batch_sync,
                batch
            )
            
            # Store results with IDs
            for batch_id, result in zip(batch_ids, results):
                await self.results_queue.put({
                    'id': batch_id,
                    'result': result
                })
    
    def _process_batch_sync(self, batch):
        """Synchronous batch processing in thread pool"""
        try:
            X = np.array(batch)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Generate explanations if available
            explanations = []
            if hasattr(self.model, 'explain'):
                explanations = self.model.explain(X)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': int(pred),
                    'probability': float(prob.max()),
                    'explanation': explanations[i] if explanations else None
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [{'error': str(e)} for _ in batch]
    
    async def start_processing(self, data_queue, num_workers=3):
        """Start multiple batch processing workers"""
        workers = [
            asyncio.create_task(self.process_batch(data_queue))
            for _ in range(num_workers)
        ]
        
        await asyncio.gather(*workers)
```

### 4. Memory Management

**Memory-Efficient Data Handling:**
```python
# src/memory_manager.py
import gc
import psutil
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory_threshold=0.8):
        self.memory_threshold = memory_threshold
    
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    @contextmanager
    def memory_monitor(self):
        """Context manager for memory monitoring"""
        initial_memory = self.get_memory_usage()
        try:
            yield
        finally:
            final_memory = self.get_memory_usage()
            if final_memory > self.memory_threshold:
                logger.warning(f"High memory usage: {final_memory:.2%}")
                self.cleanup_memory()
    
    def cleanup_memory(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        logger.info("Memory cleanup performed")
    
    def check_memory_pressure(self):
        """Check if memory pressure is high"""
        current_usage = self.get_memory_usage()
        if current_usage > self.memory_threshold:
            logger.warning(f"Memory pressure detected: {current_usage:.2%}")
            self.cleanup_memory()
            return True
        return False
```

## Scaling Strategies

### 1. Horizontal Scaling

**Auto-scaling Configuration:**
```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: xai-ids-hpa
  namespace: xai-ids
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: xai-ids-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 2. Vertical Scaling

**Resource Optimization:**
```python
# src/resource_optimizer.py
import os
import threading
import time
import psutil
from dataclasses import dataclass

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    disk_io: dict
    network_io: dict

class ResourceOptimizer:
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self.metrics_history = []
        self.running = False
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
    
    def _monitor_resources(self):
        """Background monitoring loop"""
        while self.running:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only last 100 measurements
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            # Optimize based on metrics
            self._optimize_resources(metrics)
            
            time.sleep(self.check_interval)
    
    def _collect_metrics(self):
        """Collect system resource metrics"""
        return ResourceMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_io=psutil.disk_io_counters()._asdict(),
            network_io=psutil.net_io_counters()._asdict()
        )
    
    def _optimize_resources(self, metrics):
        """Optimize resources based on metrics"""
        # Adjust worker processes based on CPU usage
        if metrics.cpu_percent > 80:
            self._reduce_workers()
        elif metrics.cpu_percent < 30:
            self._increase_workers()
        
        # Trigger memory cleanup if needed
        if metrics.memory_percent > 85:
            self._cleanup_memory()
    
    def _reduce_workers(self):
        """Reduce number of worker processes"""
        current_workers = int(os.environ.get('MAX_WORKERS', 4))
        if current_workers > 1:
            os.environ['MAX_WORKERS'] = str(current_workers - 1)
    
    def _increase_workers(self):
        """Increase number of worker processes"""
        current_workers = int(os.environ.get('MAX_WORKERS', 4))
        max_workers = psutil.cpu_count()
        if current_workers < max_workers:
            os.environ['MAX_WORKERS'] = str(current_workers + 1)
    
    def _cleanup_memory(self):
        """Trigger memory cleanup"""
        import gc
        gc.collect()
```

## Disaster Recovery

### 1. Backup Strategy

**Automated Backup System:**
```python
# src/backup_manager.py
import os
import shutil
import tarfile
import boto3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BackupManager:
    def __init__(self, config):
        self.config = config
        self.s3_client = boto3.client('s3') if config.get('s3_enabled') else None
    
    def create_backup(self):
        """Create complete system backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"xai-ids-backup-{timestamp}"
        backup_path = f"/tmp/{backup_name}"
        
        try:
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup models
            models_backup = os.path.join(backup_path, 'models')
            shutil.copytree(self.config['model_path'], models_backup)
            
            # Backup configurations
            config_backup = os.path.join(backup_path, 'configs')
            shutil.copytree(self.config['config_path'], config_backup)
            
            # Backup database
            self._backup_database(backup_path)
            
            # Create tarball
            tarball_path = f"{backup_path}.tar.gz"
            with tarfile.open(tarball_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_name)
            
            # Upload to S3 if configured
            if self.s3_client:
                self._upload_to_s3(tarball_path, f"{backup_name}.tar.gz")
            
            # Cleanup local files
            shutil.rmtree(backup_path)
            if self.config.get('keep_local_backup', False):
                return tarball_path
            else:
                os.remove(tarball_path)
                
            logger.info(f"Backup created successfully: {backup_name}")
            return backup_name
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
    
    def _backup_database(self, backup_path):
        """Backup PostgreSQL database"""
        db_backup_path = os.path.join(backup_path, 'database.sql')
        db_url = self.config['database_url']
        
        # Extract database info from URL
        # postgresql://user:password@host:port/database
        import urllib.parse
        parsed = urllib.parse.urlparse(db_url)
        
        cmd = f"pg_dump -h {parsed.hostname} -p {parsed.port} -U {parsed.username} -d {parsed.path[1:]} > {db_backup_path}"
        os.system(cmd)
    
    def _upload_to_s3(self, file_path, s3_key):
        """Upload backup to S3"""
        bucket = self.config['s3_bucket']
        try:
            self.s3_client.upload_file(file_path, bucket, s3_key)
            logger.info(f"Backup uploaded to S3: s3://{bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def restore_backup(self, backup_name):
        """Restore from backup"""
        try:
            # Download from S3 if needed
            if self.s3_client:
                local_path = f"/tmp/{backup_name}.tar.gz"
                self.s3_client.download_file(
                    self.config['s3_bucket'], 
                    f"{backup_name}.tar.gz", 
                    local_path
                )
            else:
                local_path = f"/backups/{backup_name}.tar.gz"
            
            # Extract backup
            extract_path = f"/tmp/{backup_name}"
            with tarfile.open(local_path, 'r:gz') as tar:
                tar.extractall('/tmp')
            
            # Restore models
            if os.path.exists(os.path.join(extract_path, 'models')):
                shutil.rmtree(self.config['model_path'])
                shutil.copytree(
                    os.path.join(extract_path, 'models'),
                    self.config['model_path']
                )
            
            # Restore configurations
            if os.path.exists(os.path.join(extract_path, 'configs')):
                shutil.rmtree(self.config['config_path'])
                shutil.copytree(
                    os.path.join(extract_path, 'configs'),
                    self.config['config_path']
                )
            
            # Restore database
            db_backup_file = os.path.join(extract_path, 'database.sql')
            if os.path.exists(db_backup_file):
                self._restore_database(db_backup_file)
            
            # Cleanup
            shutil.rmtree(extract_path)
            if self.s3_client:
                os.remove(local_path)
            
            logger.info(f"Restore completed successfully: {backup_name}")
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise
    
    def _restore_database(self, backup_file):
        """Restore PostgreSQL database"""
        db_url = self.config['database_url']
        import urllib.parse
        parsed = urllib.parse.urlparse(db_url)
        
        cmd = f"psql -h {parsed.hostname} -p {parsed.port} -U {parsed.username} -d {parsed.path[1:]} < {backup_file}"
        os.system(cmd)
```

### 2. Failover Mechanisms

**Automatic Failover:**
```python
# src/failover_manager.py
import time
import logging
import requests
from threading import Thread
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class ServiceEndpoint:
    url: str
    priority: int
    healthy: bool = True
    last_check: float = 0

class FailoverManager:
    def __init__(self, endpoints: List[ServiceEndpoint], check_interval=30):
        self.endpoints = sorted(endpoints, key=lambda x: x.priority)
        self.check_interval = check_interval
        self.current_endpoint = self.endpoints[0] if endpoints else None
        self.monitoring = False
    
    def start_monitoring(self):
        """Start health monitoring in background"""
        self.monitoring = True
        monitor_thread = Thread(target=self._monitor_health)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("Failover monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
    
    def _monitor_health(self):
        """Monitor endpoint health"""
        while self.monitoring:
            for endpoint in self.endpoints:
                try:
                    response = requests.get(
                        f"{endpoint.url}/health",
                        timeout=5
                    )
                    endpoint.healthy = response.status_code == 200
                    endpoint.last_check = time.time()
                    
                except requests.RequestException:
                    endpoint.healthy = False
                    endpoint.last_check = time.time()
            
            # Update current endpoint if needed
            self._update_current_endpoint()
            time.sleep(self.check_interval)
    
    def _update_current_endpoint(self):
        """Update current endpoint based on health"""
        for endpoint in self.endpoints:
            if endpoint.healthy:
                if self.current_endpoint != endpoint:
                    logger.warning(f"Failing over to: {endpoint.url}")
                    self.current_endpoint = endpoint
                break
        else:
            logger.error("No healthy endpoints available")
            self.current_endpoint = None
    
    def get_current_endpoint(self) -> Optional[ServiceEndpoint]:
        """Get current healthy endpoint"""
        return self.current_endpoint
    
    def make_request(self, path: str, **kwargs):
        """Make request with automatic fallback"""
        if not self.current_endpoint:
            raise Exception("No healthy endpoints available")
        
        try:
            url = f"{self.current_endpoint.url}{path}"
            return requests.request(url=url, **kwargs)
        except requests.RequestException as e:
            # Mark current endpoint as unhealthy and try next
            self.current_endpoint.healthy = False
            self._update_current_endpoint()
            
            if self.current_endpoint:
                return self.make_request(path, **kwargs)
            else:
                raise Exception("All endpoints failed") from e
```

### 3. Recovery Procedures

**Disaster Recovery Playbook:**
```yaml
# disaster-recovery-playbook.yaml
recovery_procedures:
  
  complete_system_failure:
    steps:
      - name: "Assess Damage"
        actions:
          - Check system logs
          - Identify root cause
          - Document incident
      
      - name: "Restore Infrastructure"
        actions:
          - Deploy fresh infrastructure
          - Configure networking
          - Set up security groups
      
      - name: "Restore Application"
        actions:
          - Pull latest backup
          - Restore database
          - Deploy application code
          - Restore model files
      
      - name: "Validate Recovery"
        actions:
          - Run health checks
          - Test API endpoints
          - Verify model predictions
          - Check monitoring systems
      
      - name: "Resume Operations"
        actions:
          - Update DNS records
          - Notify stakeholders
          - Monitor for issues
  
  database_corruption:
    steps:
      - name: "Stop Applications"
        actions:
          - Scale down to zero replicas
          - Prevent new writes
      
      - name: "Restore Database"
        actions:
          - Restore from latest backup
          - Run integrity checks
          - Update connection strings
      
      - name: "Restart Applications"
        actions:
          - Scale back up
          - Validate functionality
  
  model_corruption:
    steps:
      - name: "Switch to Backup Model"
        actions:
          - Load previous model version
          - Update model registry
          - Restart prediction services
      
      - name: "Retrain if Necessary"
        actions:
          - Trigger training pipeline
          - Validate new model
          - Deploy when ready
```

## Monitoring and Alerting

### 1. Comprehensive Monitoring

**Custom Metrics Collection:**
```python
# src/metrics_collector.py
import time
import threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps

# Define metrics
prediction_counter = Counter('xai_ids_predictions_total', 'Total predictions made')
prediction_latency = Histogram('xai_ids_prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('xai_ids_model_accuracy', 'Current model accuracy')
active_connections = Gauge('xai_ids_active_connections', 'Active connections')
memory_usage = Gauge('xai_ids_memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('xai_ids_cpu_usage_percent', 'CPU usage percentage')

class MetricsCollector:
    def __init__(self, port=9090):
        self.port = port
        self.monitoring = False
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.port)
        self.monitoring = True
        
        # Start system metrics collection
        metrics_thread = threading.Thread(target=self._collect_system_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        import psutil
        
        while self.monitoring:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage.set(memory.used)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_usage.set(cpu_percent)
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    @staticmethod
    def time_prediction(func):
        """Decorator to time prediction functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                prediction_counter.inc()
                return result
            finally:
                prediction_latency.observe(time.time() - start_time)
        return wrapper
    
    @staticmethod
    def update_model_accuracy(accuracy):
        """Update model accuracy metric"""
        model_accuracy.set(accuracy)
    
    @staticmethod
    def increment_connections():
        """Increment active connections"""
        active_connections.inc()
    
    @staticmethod
    def decrement_connections():
        """Decrement active connections"""
        active_connections.dec()
```

### 2. Alert Configuration

**AlertManager Rules:**
```yaml
# alerting-rules.yaml
groups:
- name: xai-ids-alerts
  rules:
  
  - alert: HighPredictionLatency
    expr: histogram_quantile(0.95, xai_ids_prediction_duration_seconds_bucket) > 2.0
    for: 5m
    labels:
      severity: warning
      service: xai-ids
    annotations:
      summary: "High prediction latency detected"
      description: "95th percentile prediction latency is {{ $value }}s"
  
  - alert: LowPredictionRate
    expr: rate(xai_ids_predictions_total[5m]) < 10
    for: 10m
    labels:
      severity: warning
      service: xai-ids
    annotations:
      summary: "Low prediction rate"
      description: "Current prediction rate is {{ $value }} predictions/second"
  
  - alert: ModelAccuracyDrop
    expr: xai_ids_model_accuracy < 0.85
    for: 5m
    labels:
      severity: critical
      service: xai-ids
    annotations:
      summary: "Model accuracy has dropped"
      description: "Current model accuracy is {{ $value }}"
  
  - alert: HighMemoryUsage
    expr: xai_ids_memory_usage_bytes / (1024^3) > 8
    for: 5m
    labels:
      severity: warning
      service: xai-ids
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}GB"
  
  - alert: ServiceDown
    expr: up{job="xai-ids"} == 0
    for: 1m
    labels:
      severity: critical
      service: xai-ids
    annotations:
      summary: "XAI-IDS service is down"
      description: "Service has been down for more than 1 minute"
```

### 3. Health Checks

**Comprehensive Health Check System:**
```python
# src/health_checker.py
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    response_time: float
    details: Optional[Dict] = None

class HealthChecker:
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func, timeout=5):
        """Register a health check function"""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout
        }
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check"""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found",
                response_time=0
            )
        
        check = self.checks[name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                check['func'](),
                timeout=check['timeout']
            )
            response_time = asyncio.get_event_loop().time() - start_time
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message="OK",
                response_time=response_time,
                details=result
            )
            
        except asyncio.TimeoutError:
            response_time = asyncio.get_event_loop().time() - start_time
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Timeout",
                response_time=response_time
            )
            
        except Exception as e:
            response_time = asyncio.get_event_loop().time() - start_time
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time=response_time
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        tasks = [
            self.run_check(name) 
            for name in self.checks.keys()
        ]
        
        results = await asyncio.gather(*tasks)
        return {result.name: result for result in results}
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system health"""
        if not results:
            return HealthStatus.UNHEALTHY
        
        statuses = [result.status for result in results.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

# Health check implementations
async def check_database():
    """Check database connectivity"""
    import asyncpg
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute('SELECT 1')
        await conn.close()
        return {"status": "connected"}
    except Exception as e:
        raise Exception(f"Database check failed: {e}")

async def check_redis():
    """Check Redis connectivity"""
    import aioredis
    try:
        redis = aioredis.from_url(REDIS_URL)
        await redis.ping()
        await redis.close()
        return {"status": "connected"}
    except Exception as e:
        raise Exception(f"Redis check failed: {e}")

async def check_model():
    """Check model availability"""
    import os
    model_path = os.environ.get('XAI_IDS_MODEL_PATH')
    if not os.path.exists(model_path):
        raise Exception("Model file not found")
    
    # Try loading model
    try:
        import joblib
        model = joblib.load(model_path)
        return {
            "status": "loaded",
            "model_type": str(type(model).__name__)
        }
    except Exception as e:
        raise Exception(f"Model loading failed: {e}")

async def check_disk_space():
    """Check available disk space"""
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    free_percent = (free / total) * 100
    
    if free_percent < 10:
        raise Exception(f"Low disk space: {free_percent:.1f}% free")
    
    return {
        "free_percent": free_percent,
        "free_gb": free // (1024**3),
        "total_gb": total // (1024**3)
    }
```

## Security Hardening

### 1. Container Security

**Secure Dockerfile:**
```dockerfile
# Multi-stage build for security
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r xai-ids && useradd -r -g xai-ids xai-ids

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY --chown=xai-ids:xai-ids . /app
WORKDIR /app

# Set security headers
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Remove unnecessary packages
RUN apt-get autoremove -y && apt-get clean

# Set file permissions
RUN chmod -R 755 /app && \
    chmod -R 644 /app/configs && \
    chmod 750 /app/src

# Switch to non-root user
USER xai-ids

# Security scanning
LABEL security.scan="enabled"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "src/api_server.py"]
```

### 2. Network Security

**Network Policies:**
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xai-ids-network-policy
  namespace: xai-ids
spec:
  podSelector:
    matchLabels:
      app: xai-ids
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  
  - to:
    - namespaceSelector:
        matchLabels:
          name: redis
    ports:
    - protocol: TCP
      port: 6379
  
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

## Deployment Checklist

### Pre-Deployment
- [ ] Environment configuration validated
- [ ] Security credentials rotated
- [ ] Database migrations tested
- [ ] Model files validated and uploaded
- [ ] Configuration files reviewed
- [ ] Resource requirements calculated
- [ ] Monitoring dashboards prepared
- [ ] Backup procedures tested
- [ ] Disaster recovery plan reviewed

### Deployment
- [ ] Infrastructure provisioned
- [ ] Application deployed
- [ ] Database connections verified
- [ ] Model loading confirmed
- [ ] Health checks passing
- [ ] Monitoring active
- [ ] Logging configured
- [ ] Security scans completed

### Post-Deployment
- [ ] Performance baseline established
- [ ] Alert thresholds configured
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Incident response procedures updated
- [ ] Backup schedule activated
- [ ] Security audit completed
- [ ] Stakeholders notified

## Troubleshooting Guide

### Common Issues

**1. Model Loading Failures**
```bash
# Check model file integrity
python -c "import joblib; model = joblib.load('model.pkl'); print('Model loaded successfully')"

# Verify model compatibility
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

# Check file permissions
ls -la /path/to/model/
```

**2. Database Connection Issues**
```bash
# Test database connectivity
pg_isready -h localhost -p 5432

# Check connection limits
psql -c "SELECT * FROM pg_stat_activity;"

# Verify credentials
psql postgresql://user:pass@host:port/database -c "SELECT 1;"
```

**3. Memory Issues**
```bash
# Monitor memory usage
free -h
htop

# Check application memory
ps aux | grep xai-ids

# Monitor garbage collection
python -c "import gc; print(f'GC stats: {gc.get_stats()}')"
```

**4. Performance Issues**
```bash
# Check CPU usage
top -p $(pgrep -f xai-ids)

# Monitor I/O
iotop -p $(pgrep -f xai-ids)

# Check network latency
ping -c 5 database-host
```

This completes the comprehensive deployment guide for the XAI Intrusion Detection System, covering everything from basic installations to enterprise-grade production deployments with high availability, monitoring, security, and disaster recovery procedures.
