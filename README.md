# DeepfakeBench Dependency & Environment Issues: Complete Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Docker Environment Challenges](#docker-environment-challenges)
3. [Python Dependencies Issues](#python-dependencies-issues)
4. [CUDA & PyTorch Compatibility](#cuda--pytorch-compatibility)
5. [Database Integration Dependencies](#database-integration-dependencies)
6. [System-Level Dependencies](#system-level-dependencies)
7. [Memory Management Dependencies](#memory-management-dependencies)
8. [Dataset Loading Dependencies](#dataset-loading-dependencies)
9. [Model Weight Dependencies](#model-weight-dependencies)
10. [Configuration File Dependencies](#configuration-file-dependencies)
11. [File Path & Volume Mount Issues](#file-path--volume-mount-issues)
12. [Network & Port Dependencies](#network--port-dependencies)
13. [Build & Compilation Issues](#build--compilation-issues)
14. [Version Compatibility Matrix](#version-compatibility-matrix)
15. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview

This document details all the dependency and environment issues encountered during the DeepfakeBench migration from H100 to RTX 4090, including specific fixes, version conflicts, and system-level challenges that required resolution.

### Key Challenges Summary
- **Docker Base Image**: PyTorch version compatibility issues
- **Python Dependencies**: Missing packages, version conflicts
- **CUDA Compatibility**: RTX 4090 specific requirements
- **Database Integration**: PostgreSQL setup and connection issues
- **Memory Management**: RTX 4090 memory constraints
- **File System**: Path resolution and volume mounting
- **Network Configuration**: Port conflicts and service dependencies

---

## Original vs Modified Requirements

### Original DeepfakeBench Requirements

#### Original Dockerfile
```dockerfile
# Original Dockerfile (DeepfakeBench_org/DeepfakeBench_test-main/Dockerfile)
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

LABEL maintainer="Deepfake"

# Install dependencies outside of the base image
RUN DEBIAN_FRONTEND=noninteractive apt update && \
	apt install -y --no-install-recommends automake \
    build-essential  \
    ca-certificates  \
    libfreetype6-dev  \
    libtool  \
    pkg-config  \
    python-dev  \
    python-distutils-extra \
    python3.7-dev  \
    python3-pip \
    cmake \
	&& \
    rm -rf /var/lib/apt/lists/* \
    && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 0  \
    && \
    python3.7 -m pip install pip --upgrade 

WORKDIR /

# Install Python dependencies
RUN pip install --no-cache-dir certifi setuptools \
    && \
    pip --no-cache-dir install dlib==19.24.0\
    imageio==2.9.0\
    imgaug==0.4.0\
    scipy==1.7.3\
    seaborn==0.11.2\
    pyyaml==6.0\
    imutils==0.5.4\
    opencv-python==4.6.0.66\
    scikit-image==0.19.2\
    scikit-learn==1.0.2\
    efficientnet-pytorch==0.7.1\
    timm==0.6.12\
    segmentation-models-pytorch==0.3.2\
    torchtoolbox==0.1.8.2\
    tensorboard==2.10.1\
    pip install setuptools==59.5.0 \
    pip install loralib \
    pip install pytorchvideo \
    pip install einops \
    pip install transformers \
    pip install filterpy \
    pip install simplejson \
    pip install kornia \
    pip install git+https://github.com/openai/CLIP.git

ENV MODEL_NAME=deepfakebench

# Expose port
EXPOSE 6000
```

#### Original Python Version
- **Python**: 3.7.2
- **PyTorch**: 1.12.0
- **CUDA**: 11.3
- **cuDNN**: 8

#### Original Package Versions
```txt
# Original requirements (inferred from Dockerfile)
dlib==19.24.0
imageio==2.9.0
imgaug==0.4.0
scipy==1.7.3
seaborn==0.11.2
pyyaml==6.0
imutils==0.5.4
opencv-python==4.6.0.66
scikit-image==0.19.2
scikit-learn==1.0.2
efficientnet-pytorch==0.7.1
timm==0.6.12
segmentation-models-pytorch==0.3.2
torchtoolbox==0.1.8.2
tensorboard==2.10.1
setuptools==59.5.0
loralib
pytorchvideo
einops
transformers
filterpy
simplejson
kornia
git+https://github.com/openai/CLIP.git
```

### Modified Requirements (Current Implementation)

#### Modified Dockerfile
```dockerfile
# Current Dockerfile (modified for RTX 4090)
FROM nvcr.io/nvidia/pytorch:23.08-py3

LABEL maintainer="Deepfake"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    python3-dev \
    cmake \
    libpq-dev \
    libopencv-dev \
    libdlib-dev \
    libfreetype6-dev \
    libtool \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    dash \
    plotly \
    dash-bootstrap-components \
    python-dotenv \
    scikit-learn \
    dlib==19.24.0 \
    imageio==2.9.0 \
    imgaug==0.4.0 \
    scipy==1.7.3 \
    seaborn==0.11.2 \
    pyyaml==6.0 \
    imutils==0.5.4 \
    opencv-python==4.6.0.66 \
    scikit-image==0.19.2 \
    efficientnet-pytorch==0.7.1 \
    timm==0.6.12 \
    segmentation-models-pytorch==0.3.2 \
    torchtoolbox==0.1.8.2 \
    tensorboard==2.10.1 \
    setuptools==59.5.0 \
    loralib \
    pytorchvideo \
    einops \
    transformers \
    filterpy \
    simplejson \
    kornia \
    git+https://github.com/openai/CLIP.git

ENV MODEL_NAME=deepfakebench

# Expose ports
EXPOSE 8050 5432
```

#### Modified Python Version
- **Python**: 3.8+ (from nvcr.io/nvidia/pytorch:23.08-py3)
- **PyTorch**: 2.0.1+ (from nvcr.io/nvidia/pytorch:23.08-py3)
- **CUDA**: 12.1+ (from nvcr.io/nvidia/pytorch:23.08-py3)
- **cuDNN**: 8.9+ (from nvcr.io/nvidia/pytorch:23.08-py3)

#### Added Package Versions
```txt
# New packages added for database and dashboard
psycopg2-binary==2.9.7
dash==2.14.1
plotly==5.17.0
dash-bootstrap-components==1.5.0
python-dotenv==1.0.0
scikit-learn==1.3.0  # Updated from 1.0.2
```

### Key Changes Summary

#### 1. Docker Base Image Changes
| Component | Original | Modified | Reason |
|-----------|----------|----------|---------|
| Base Image | `pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel` | `nvcr.io/nvidia/pytorch:23.08-py3` | RTX 4090 compatibility |
| Python Version | 3.7.2 | 3.8+ | Better PyTorch 2.0+ support |
| PyTorch Version | 1.12.0 | 2.0.1+ | RTX 4090 optimization |
| CUDA Version | 11.3 | 12.1+ | RTX 4090 support |
| cuDNN Version | 8.0 | 8.9+ | Latest optimization |

#### 2. System Dependencies Changes
| Package | Original | Modified | Reason |
|---------|----------|----------|---------|
| PostgreSQL | ❌ Not included | ✅ Added | Database integration |
| Build Tools | Basic | Enhanced | Better compilation |
| Python Dev | python3.7-dev | python3-dev | Generic Python dev |
| Database Headers | ❌ Missing | ✅ libpq-dev | PostgreSQL support |

#### 3. Python Package Changes
| Package | Original | Modified | Reason |
|---------|----------|----------|---------|
| scikit-learn | 1.0.2 | 1.3.0 | Better metrics calculation |
| psycopg2-binary | ❌ Missing | ✅ 2.9.7 | Database connectivity |
| dash | ❌ Missing | ✅ 2.14.1 | Web dashboard |
| plotly | ❌ Missing | ✅ 5.17.0 | Interactive charts |
| dash-bootstrap-components | ❌ Missing | ✅ 1.5.0 | UI components |
| python-dotenv | ❌ Missing | ✅ 1.0.0 | Environment variables |

#### 4. Port Configuration Changes
| Port | Original | Modified | Reason |
|------|----------|----------|---------|
| TensorBoard | 6000 | 6000 | Kept same |
| Dashboard | ❌ None | ✅ 8050 | New web dashboard |
| PostgreSQL | ❌ None | ✅ 5432 | Database service |

#### 5. Environment Variables Changes
| Variable | Original | Modified | Reason |
|----------|----------|----------|---------|
| MODEL_NAME | deepfakebench | deepfakebench | Kept same |
| PYTHONPATH | ❌ Not set | ✅ /workspace | Path resolution |
| PYTHONUNBUFFERED | ❌ Not set | ✅ 1 | Better logging |
| DB_HOST | ❌ Not set | ✅ localhost | Database config |
| DB_PORT | ❌ Not set | ✅ 5432 | Database config |
| DB_NAME | ❌ Not set | ✅ deepfake_bench | Database config |
| DB_USER | ❌ Not set | ✅ deepfake_user | Database config |
| DB_PASSWORD | ❌ Not set | ✅ deepfake_pass | Database config |

### Configuration File Changes

#### Original Configuration Structure
```yaml
# Original config structure (minimal)
log_dir: /mntcephfs/lab_data/yuanxinhang/benchmark_results/logs_analysis/fwa
pretrained: ./training/pretrained/xception-b5690688.pth
model_name: fwa
backbone_name: xception
# ... basic config only
```

#### Modified Configuration Structure
```yaml
# Modified config structure (enhanced)
log_dir: /mntcephfs/lab_data/yuanxinhang/benchmark_results/logs_analysis/fwa
pretrained: ./training/pretrained/xception-b5690688.pth
model_name: fwa
backbone_name: xception
# ... original config ...

# NEW: Database integration
dataset_json_folder: /workspace/preprocessing/dataset_json
rgb_dir: datasets/rgb

# NEW: Comprehensive label dictionary
label_dict:
  DFD_real: 0
  DFD_fake: 1
  FF-real: 0
  FF-fake: 1
  FF-DF: 1
  FF-F2F: 1
  FF-FS: 1
  FF-NT: 1
  Celeb-DF-v1_real: 0
  Celeb-DF-v1_fake: 1
  # ... all dataset labels

# NEW: Memory optimization
train_batchSize: 16  # Reduced from 32
test_batchSize: 32   # Reduced from 64
workers: 2           # Reduced from 8
```

### File Structure Changes

#### Original File Structure
```
DeepfakeBench_test-main/
├── training/
│   ├── train.py
│   ├── test.py
│   └── config/
├── preprocessing/
├── datasets/
└── Dockerfile
```

#### Modified File Structure
```
DeepfakeBench_test/
├── training/                    # Original structure
├── preprocessing/               # Original structure
├── datasets/                    # Original structure
├── test_script/                 # NEW: Enhanced testing scripts
│   ├── test_models_hybrid_cached.py
│   ├── test_all_models_on_dataset.sh
│   ├── test_single_model_on_all_datasets.sh
│   └── test_comprehensive_all_models_all_datasets.sh
├── simple_dashboard.py          # NEW: Web dashboard
├── test_with_db.py             # NEW: Database-integrated testing
├── database_setup.py           # NEW: Database initialization
├── setup.sh                    # NEW: Automated setup
├── automated_setup.sh          # NEW: One-command setup
├── complete_setup.sh           # NEW: Interactive setup
├── verify_setup.sh             # NEW: Setup verification
├── config.env                  # NEW: Environment variables
├── Dockerfile                  # MODIFIED: RTX 4090 optimized
├── ISSUE.md                    # NEW: Complete issue documentation
├── DEP_ISSUE.md                # NEW: Dependency documentation
├── GETTING_STARTED.md          # NEW: User guide
└── README.md                   # MODIFIED: Updated documentation
```

### Memory Management Changes

#### Original Memory Configuration
```yaml
# Original memory settings (H100 optimized)
train_batchSize: 32
test_batchSize: 64
workers: 8
frame_num: {'train': 32, 'test': 32}
```

#### Modified Memory Configuration
```yaml
# Modified memory settings (RTX 4090 optimized)
train_batchSize: 16  # Reduced for RTX 4090
test_batchSize: 32   # Reduced for RTX 4090
workers: 2           # Reduced to prevent crashes
frame_num: {'train': 32, 'test': 32}  # Kept same
```

#### New Memory Management Features
```python
# NEW: Hybrid memory management
def determine_approach(dataset_name, available_ram_gb):
    estimated_memory_gb = estimate_dataset_memory_gb(num_samples)
    if estimated_memory_gb < memory_threshold_gb:
        return "IN-MEMORY"
    else:
        return "MEMORY-OPTIMIZED"

# NEW: Memory monitoring
def get_memory_usage():
    system_ram = psutil.virtual_memory()
    available_ram_gb = system_ram.available / (1024**3)
    return available_ram_gb

# NEW: GPU memory clearing
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
```

### Database Integration Changes

#### Original: No Database
- No database integration
- Results only printed to console
- No result persistence
- No historical tracking

#### Modified: Full Database Integration
```python
# NEW: Database schema
CREATE TABLE test_results (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    accuracy FLOAT,
    auc FLOAT,
    eer FLOAT,
    ap FLOAT,
    video_auc FLOAT,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, dataset_name)
);

# NEW: Database connection
def save_to_database(model_name, dataset_name, status, metrics):
    conn = psycopg2.connect(
        host="localhost", port="5432",
        database="deepfake_bench",
        user="deepfake_user",
        password="deepfake_pass"
    )
    # ... save results
```

### Dashboard Integration Changes

#### Original: No Dashboard
- No web interface
- No result visualization
- No interactive charts

#### Modified: Full Dashboard Integration
```python
# NEW: Dash-based web dashboard
import dash
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# NEW: Interactive charts
@app.callback(
    Output('results-graph', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_graph(selected_metric):
    # ... create interactive charts
```

### Testing Framework Changes

#### Original: Basic Testing
```python
# Original testing (training/test.py)
def test_epoch(model, test_data_loaders):
    for key in test_data_loaders.keys():
        predictions, labels, features = test_one_dataset(model, test_data_loaders[key])
        metrics = get_test_metrics(predictions, labels)
        print(f"dataset: {key}")
        for k, v in metrics.items():
            print(f"{k}: {v}")
```

#### Modified: Enhanced Testing
```python
# NEW: Database-integrated testing (test_with_db.py)
def save_to_database(model_name, dataset_name, status, metrics):
    # Save to PostgreSQL database
    # Include error handling
    # Track success/failure status

# NEW: Hybrid memory management testing (test_models_hybrid_cached.py)
def test_models_with_hybrid_approach(models, dataset, batch_size):
    # Automatic memory approach selection
    # In-memory for small datasets
    # Memory-optimized for large datasets
    # Results saved to database
```

### Error Handling Changes

#### Original: Basic Error Handling
```python
# Original: Minimal error handling
try:
    result = some_operation()
except Exception as e:
    print(f"Error: {e}")
```

#### Modified: Comprehensive Error Handling
```python
# NEW: Robust error handling
def calculate_metrics(predictions, labels, dataset_name):
    metrics = {}
    
    # AUC calculation with error handling
    try:
        auc = roc_auc_score(labels, predictions)
        metrics['AUC'] = float(auc)
    except Exception as e:
        print(f"⚠️  AUC calculation failed: {e}")
        metrics['AUC'] = 0.0
    
    # EER calculation with error handling
    try:
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        metrics['EER'] = float(eer)
    except Exception as e:
        print(f"⚠️  EER calculation failed: {e}")
        metrics['EER'] = 1.0
    
    return metrics
```

### Summary of All Changes

#### What Was Added
1. **Database Integration**: PostgreSQL + result storage
2. **Web Dashboard**: Dash-based interactive dashboard
3. **Memory Management**: Hybrid in-memory/memory-optimized approach
4. **Error Handling**: Comprehensive error handling and recovery
5. **Automated Setup**: One-command setup and verification
6. **Enhanced Testing**: Database-integrated testing framework
7. **Documentation**: Complete issue and dependency documentation
8. **Configuration Management**: Universal config fixer and validation

#### What Was Modified
1. **Docker Base Image**: H100 → RTX 4090 optimized
2. **Python Version**: 3.7.2 → 3.8+
3. **PyTorch Version**: 1.12.0 → 2.0.1+
4. **CUDA Version**: 11.3 → 12.1+
5. **Batch Sizes**: Reduced for RTX 4090 memory constraints
6. **Workers**: Reduced to prevent crashes
7. **Port Configuration**: Added dashboard and database ports
8. **File Structure**: Enhanced with new testing and setup scripts

#### What Was Removed
1. **H100-specific optimizations**: Replaced with RTX 4090 optimizations
2. **Outdated dependencies**: Updated to compatible versions
3. **Hardcoded paths**: Replaced with environment-based configuration

This comprehensive comparison shows exactly what changed from the original DeepfakeBench to our RTX 4090 optimized version, providing a complete technical migration record.

---

## Docker Environment Challenges

### Issue 1: Base Image Selection
**Problem**: Original Dockerfile used outdated PyTorch version incompatible with RTX 4090
**Original Configuration**:
```dockerfile
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
```

**Issues Encountered**:
- CUDA 11.3 not optimal for RTX 4090
- PyTorch 1.12.0 missing newer features
- Incompatible with newer NVIDIA drivers

**Solution Implemented**:
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.08-py3
```

**Why This Works**:
- CUDA 12.2 support for RTX 4090
- PyTorch 2.0+ with better memory management
- Optimized for newer NVIDIA hardware

### Issue 2: Docker Build Context
**Problem**: Build failures due to missing build context and large file transfers
**Symptoms**:
```
ERROR: failed to solve: failed to compute cache key: failed to calculate checksum
```

**Solutions**:
1. **Multi-stage Build**: Separate dependency installation from application code
2. **Layer Caching**: Optimize Dockerfile layer ordering
3. **Build Context**: Use `.dockerignore` to exclude unnecessary files

**Final Dockerfile Structure**:
```dockerfile
# Stage 1: Base image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Stage 2: System dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    dash \
    plotly \
    dash-bootstrap-components \
    python-dotenv \
    scikit-learn

# Stage 4: Application setup
WORKDIR /workspace
EXPOSE 8050 5432
```

### Issue 3: Container Resource Limits
**Problem**: Container running out of memory during training
**Symptoms**:
```
Killed: 9
```

**Solutions**:
1. **Memory Limits**: Set appropriate Docker memory limits
2. **Shared Memory**: Increase shared memory size
3. **GPU Memory**: Proper GPU memory allocation

**Docker Run Command**:
```bash
docker run --gpus all \
    --shm-size=64g \
    --memory=60g \
    --memory-swap=60g \
    -itd -v /path/to/datasets:/workspace/datasets \
    -p 8050:8050 -p 5432:5432 \
    --name deepfakebench-ngc \
    deepfakebench-ngc
```

---

## Python Dependencies Issues

### Issue 1: Missing Core Packages
**Problem**: Original requirements missing essential packages for database and dashboard
**Missing Packages**:
- `psycopg2-binary` (PostgreSQL adapter)
- `dash` (Web dashboard)
- `plotly` (Interactive charts)
- `dash-bootstrap-components` (UI components)
- `python-dotenv` (Environment variables)
- `scikit-learn` (Metrics calculation)

**Installation Commands**:
```bash
pip install --no-cache-dir \
    psycopg2-binary==2.9.7 \
    dash==2.14.1 \
    plotly==5.17.0 \
    dash-bootstrap-components==1.5.0 \
    python-dotenv==1.0.0 \
    scikit-learn==1.3.0
```

### Issue 2: Version Conflicts
**Problem**: Package version conflicts causing import errors
**Specific Conflicts**:
- `torch` vs `torchvision` version mismatch
- `numpy` version incompatibility
- `opencv-python` vs `opencv-contrib-python`

**Resolution Strategy**:
```bash
# Create clean environment
conda create -n deepfakebench python=3.8
conda activate deepfakebench

# Install PyTorch first (most critical)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install other packages with specific versions
pip install numpy==1.24.3
pip install opencv-python==4.8.0.74
pip install scikit-learn==1.3.0
```

### Issue 3: Import Path Issues
**Problem**: Module import failures due to Python path issues
**Symptoms**:
```
ModuleNotFoundError: No module named 'training'
ImportError: cannot import name 'DeepfakeAbstractBaseDataset'
```

**Solutions**:
1. **PYTHONPATH**: Set proper Python path in Docker
2. **Working Directory**: Ensure correct working directory
3. **Relative Imports**: Fix relative import statements

**Docker Environment Variables**:
```dockerfile
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
```

### Issue 4: Package Installation Failures
**Problem**: Some packages failing to install due to compilation issues
**Specific Failures**:
- `dlib` compilation errors
- `psycopg2` missing PostgreSQL headers
- `opencv-python` missing system libraries

**Solutions**:
```bash
# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libopencv-dev \
    libdlib-dev \
    build-essential \
    cmake

# Install Python packages with specific flags
RUN pip install --no-cache-dir \
    --no-binary=psycopg2 psycopg2-binary \
    --no-binary=dlib dlib==19.24.0
```

---

## CUDA & PyTorch Compatibility

### Issue 1: CUDA Version Mismatch
**Problem**: PyTorch compiled for different CUDA version than system
**System CUDA**: 12.2
**PyTorch CUDA**: 11.8 (incompatible)

**Error Messages**:
```
RuntimeError: CUDA runtime error: no kernel image is available for execution
```

**Solution**:
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: GPU Memory Allocation
**Problem**: CUDA out of memory errors on RTX 4090
**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. **Memory Clearing**: Regular GPU memory cleanup
2. **Batch Size Reduction**: Reduce batch sizes for RTX 4090
3. **Gradient Accumulation**: Use gradient accumulation for effective larger batches

**Memory Management Code**:
```python
import torch
import gc

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Use after each batch
for batch in dataloader:
    # Process batch
    clear_gpu_memory()
```

### Issue 3: cuDNN Compatibility
**Problem**: cuDNN version mismatch causing convolution errors
**Error Messages**:
```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
```

**Solution**:
```bash
# Install compatible cuDNN version
pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Issue 4: Tensor Core Utilization
**Problem**: RTX 4090 not utilizing Tensor Cores efficiently
**Solutions**:
1. **Mixed Precision**: Enable automatic mixed precision
2. **Data Types**: Use appropriate data types (float16, bfloat16)
3. **Model Optimization**: Use optimized model implementations

**Mixed Precision Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Database Integration Dependencies

### Issue 1: PostgreSQL Installation
**Problem**: PostgreSQL not available in base PyTorch image
**Solution**:
```dockerfile
# Install PostgreSQL client
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python PostgreSQL adapter
RUN pip install psycopg2-binary==2.9.7
```

### Issue 2: Database Connection Issues
**Problem**: Connection failures between container and database
**Error Messages**:
```
psycopg2.OperationalError: could not connect to server
```

**Solutions**:
1. **Network Configuration**: Proper Docker networking
2. **Connection Parameters**: Correct host, port, credentials
3. **Firewall**: Ensure ports are accessible

**Connection Configuration**:
```python
import psycopg2
import os

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'deepfake_bench'),
        user=os.getenv('DB_USER', 'deepfake_user'),
        password=os.getenv('DB_PASSWORD', 'deepfake_pass')
    )
```

### Issue 3: Database Schema Dependencies
**Problem**: Missing database tables and schema
**Solution**:
```python
# Database schema creation
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS test_results (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    config_file VARCHAR(255),
    weights_file VARCHAR(255),
    accuracy FLOAT,
    auc FLOAT,
    eer FLOAT,
    ap FLOAT,
    video_auc FLOAT,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, dataset_name)
);
"""
```

### Issue 4: Database Performance Issues
**Problem**: Slow database operations affecting training/testing
**Solutions**:
1. **Connection Pooling**: Use connection pooling
2. **Batch Inserts**: Batch database operations
3. **Indexing**: Add appropriate database indexes

**Connection Pooling**:
```python
from psycopg2 import pool

class DatabaseManager:
    def __init__(self):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host='localhost',
            port='5432',
            database='deepfake_bench',
            user='deepfake_user',
            password='deepfake_pass'
        )
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        self.connection_pool.putconn(conn)
```

---

## System-Level Dependencies

### Issue 1: Missing System Libraries
**Problem**: Missing system libraries for Python package compilation
**Missing Libraries**:
- `libpq-dev` (PostgreSQL development headers)
- `libopencv-dev` (OpenCV development headers)
- `libdlib-dev` (dlib development headers)
- `build-essential` (C/C++ compiler)

**Solution**:
```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libpq-dev \
    libopencv-dev \
    libdlib-dev \
    libfreetype6-dev \
    libtool \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
```

### Issue 2: File System Permissions
**Problem**: Permission issues with dataset access and log writing
**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
1. **User Permissions**: Set appropriate user in Docker
2. **Volume Permissions**: Proper volume mounting with correct permissions
3. **Directory Creation**: Ensure directories exist with correct permissions

**Docker User Configuration**:
```dockerfile
# Create user with appropriate permissions
RUN useradd -m -s /bin/bash deepfake && \
    usermod -aG sudo deepfake

USER deepfake
WORKDIR /workspace
```

### Issue 3: Shared Memory Issues
**Problem**: Insufficient shared memory for multiprocessing
**Symptoms**:
```
OSError: [Errno 28] No space left on device
```

**Solution**:
```bash
# Increase shared memory size in Docker
docker run --shm-size=64g ...
```

### Issue 4: Process Limits
**Problem**: System process limits causing DataLoader worker failures
**Solutions**:
1. **Worker Reduction**: Reduce number of DataLoader workers
2. **Process Limits**: Increase system process limits
3. **Memory Limits**: Set appropriate memory limits per process

**DataLoader Configuration**:
```python
# Reduce workers for stability
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2,  # Reduced from 8
    pin_memory=True,
    persistent_workers=False
)
```

---

## Memory Management Dependencies

### Issue 1: RTX 4090 Memory Constraints
**Problem**: 24GB VRAM insufficient for large batch sizes
**Original Configuration**: Batch size 32-64
**RTX 4090 Limit**: Batch size 8-16 for large datasets

**Memory Usage Analysis**:
```python
def estimate_memory_usage(batch_size, image_size, model_size):
    # Image memory: batch_size * channels * height * width * 4 bytes
    image_memory = batch_size * 3 * image_size * image_size * 4
    
    # Model memory: parameters * 4 bytes (float32)
    model_memory = model_size * 4
    
    # Gradient memory: same as model memory
    gradient_memory = model_memory
    
    # Total memory
    total_memory = image_memory + model_memory + gradient_memory
    return total_memory / (1024**3)  # Convert to GB
```

### Issue 2: System RAM Management
**Problem**: Large datasets consuming too much system RAM
**Solutions**:
1. **Hybrid Loading**: In-memory for small datasets, batch loading for large
2. **Memory Monitoring**: Real-time memory usage tracking
3. **Garbage Collection**: Regular memory cleanup

**Memory Monitoring**:
```python
import psutil
import torch

def get_memory_usage():
    # System RAM
    system_ram = psutil.virtual_memory()
    available_ram_gb = system_ram.available / (1024**3)
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_available = gpu_total - gpu_memory
    else:
        gpu_available = 0
    
    return available_ram_gb, gpu_available
```

### Issue 3: Memory Leak Prevention
**Problem**: Memory leaks in DataLoader and model inference
**Solutions**:
1. **Context Managers**: Proper resource management
2. **Memory Clearing**: Regular memory cleanup
3. **Worker Management**: Proper DataLoader worker cleanup

**Memory Leak Prevention**:
```python
import gc
import torch

class MemoryManager:
    @staticmethod
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @staticmethod
    def monitor_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

---

## Dataset Loading Dependencies

### Issue 1: JSON File Structure
**Problem**: Inconsistent JSON file structure across datasets
**Original Structure**: Flat structure
**Required Structure**: Nested structure with proper labels

**JSON Structure Fix**:
```python
# Original problematic structure
{
    "test": {
        "image": [...],
        "label": [...]
    }
}

# Fixed structure
{
    "FaceForensics++": {
        "FF-real": {
            "image": [...],
            "label": "FF-real"
        },
        "FF-DF": {
            "image": [...],
            "label": "FF-DF"
        }
    }
}
```

### Issue 2: Label Dictionary Dependencies
**Problem**: Missing label mappings in configuration files
**Required Labels**:
```yaml
label_dict:
  DFD_real: 0
  DFD_fake: 1
  FF-real: 0
  FF-fake: 1
  FF-DF: 1
  FF-F2F: 1
  FF-FS: 1
  FF-NT: 1
  Celeb-DF-v1_real: 0
  Celeb-DF-v1_fake: 1
  # ... more labels
```

### Issue 3: File Path Dependencies
**Problem**: Hardcoded paths causing file not found errors
**Solutions**:
1. **Absolute Paths**: Convert all paths to absolute
2. **Environment Variables**: Use environment variables for paths
3. **Volume Mounting**: Proper Docker volume mounting

**Path Configuration**:
```python
import os

# Environment-based path configuration
DATASET_ROOT = os.getenv('DATASET_ROOT', '/workspace/datasets')
CONFIG_ROOT = os.getenv('CONFIG_ROOT', '/workspace/preprocessing/dataset_json')
WEIGHTS_ROOT = os.getenv('WEIGHTS_ROOT', '/workspace/training/weights')
```

### Issue 4: Image Loading Dependencies
**Problem**: Image loading failures due to format/corruption issues
**Solutions**:
1. **Error Handling**: Robust image loading with fallbacks
2. **Format Support**: Support multiple image formats
3. **Validation**: Image validation before processing

**Robust Image Loading**:
```python
from PIL import Image
import cv2
import numpy as np

def load_image_robust(image_path):
    try:
        # Try PIL first
        image = Image.open(image_path)
        image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        try:
            # Fallback to OpenCV
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e2:
            print(f"Failed to load image {image_path}: {e2}")
            # Return a default image or skip
            return None
```

---

## Model Weight Dependencies

### Issue 1: Missing Pre-trained Weights
**Problem**: Model weights not available or corrupted
**Required Weights**:
- Xception backbone weights
- FWA model weights
- All 13 model weights

**Weight Download Script**:
```bash
#!/bin/bash
# Download all required weights
WEIGHTS_DIR="/workspace/training/weights"
mkdir -p $WEIGHTS_DIR

# Download Xception weights
wget -O $WEIGHTS_DIR/xception-b5690688.pth \
    "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/xception-b5690688.pth"

# Download all model weights
wget -O $WEIGHTS_DIR/all_weights.zip \
    "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/pretrained.zip"

cd $WEIGHTS_DIR
unzip all_weights.zip
```

### Issue 2: Weight File Format Issues
**Problem**: Incompatible weight file formats
**Solutions**:
1. **Format Conversion**: Convert between PyTorch formats
2. **Version Compatibility**: Ensure PyTorch version compatibility
3. **State Dict Loading**: Proper state dict loading

**Weight Loading**:
```python
import torch

def load_weights_safely(model, weight_path):
    try:
        # Load state dict
        state_dict = torch.load(weight_path, map_location='cpu')
        
        # Handle different weight formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded weights from {weight_path}")
        return True
    except Exception as e:
        print(f"Failed to load weights from {weight_path}: {e}")
        return False
```

### Issue 3: Model Architecture Mismatch
**Problem**: Weight file doesn't match model architecture
**Solutions**:
1. **Architecture Verification**: Verify model architecture
2. **Partial Loading**: Load compatible layers only
3. **Weight Mapping**: Map weights to correct layers

**Architecture Verification**:
```python
def verify_model_architecture(model, weight_path):
    # Load weights
    state_dict = torch.load(weight_path, map_location='cpu')
    
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Check compatibility
    compatible_layers = []
    incompatible_layers = []
    
    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                compatible_layers.append(name)
            else:
                incompatible_layers.append(name)
        else:
            incompatible_layers.append(name)
    
    print(f"Compatible layers: {len(compatible_layers)}")
    print(f"Incompatible layers: {len(incompatible_layers)}")
    
    return compatible_layers, incompatible_layers
```

---

## Configuration File Dependencies

### Issue 1: Missing Configuration Parameters
**Problem**: Configuration files missing required parameters
**Missing Parameters**:
- `dataset_json_folder`
- `rgb_dir`
- `label_dict`
- `workers`
- `batch_size`

**Universal Config Fixer**:
```python
import yaml
import os

def fix_config_file(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add missing parameters
    if 'dataset_json_folder' not in config:
        config['dataset_json_folder'] = '/workspace/preprocessing/dataset_json'
    
    if 'rgb_dir' not in config:
        config['rgb_dir'] = 'datasets/rgb'
    
    if 'label_dict' not in config:
        config['label_dict'] = {
            'DFD_real': 0, 'DFD_fake': 1,
            'FF-real': 0, 'FF-fake': 1,
            'FF-DF': 1, 'FF-F2F': 1, 'FF-FS': 1, 'FF-NT': 1,
            'Celeb-DF-v1_real': 0, 'Celeb-DF-v1_fake': 1,
            'Celeb-DF-v2_real': 0, 'Celeb-DF-v2_fake': 1,
            'DFDC_real': 0, 'DFDC_fake': 1,
            'DFDCP_real': 0, 'DFDCP_fake': 1,
            'UADFV_real': 0, 'UADFV_fake': 1,
            'FaceShifter_real': 0, 'FaceShifter_fake': 1
        }
    
    # Save fixed config
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
```

### Issue 2: YAML Format Issues
**Problem**: Invalid YAML syntax causing parsing errors
**Common Issues**:
- Incorrect indentation
- Missing quotes around strings
- Invalid data types

**YAML Validation**:
```python
import yaml

def validate_yaml_file(file_path):
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        print(f"✅ {file_path} is valid YAML")
        return True
    except yaml.YAMLError as e:
        print(f"❌ {file_path} has YAML error: {e}")
        return False
```

### Issue 3: Environment Variable Dependencies
**Problem**: Configuration files using environment variables not set
**Solutions**:
1. **Default Values**: Provide default values for all variables
2. **Environment Setup**: Set all required environment variables
3. **Validation**: Validate all required variables are set

**Environment Variable Setup**:
```bash
# Set all required environment variables
export DATASET_ROOT=/workspace/datasets
export CONFIG_ROOT=/workspace/preprocessing/dataset_json
export WEIGHTS_ROOT=/workspace/training/weights
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=deepfake_bench
export DB_USER=deepfake_user
export DB_PASSWORD=deepfake_pass
```

---

## File Path & Volume Mount Issues

### Issue 1: Volume Mounting Problems
**Problem**: Dataset files not accessible in container
**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions**:
1. **Absolute Paths**: Use absolute paths for volume mounting
2. **Permission Mapping**: Map host user to container user
3. **Volume Verification**: Verify volume mounting

**Volume Mounting**:
```bash
# Proper volume mounting with permissions
docker run --gpus all \
    -v /absolute/path/to/datasets:/workspace/datasets:ro \
    -v /absolute/path/to/configs:/workspace/preprocessing/dataset_json:ro \
    -v /absolute/path/to/weights:/workspace/training/weights:ro \
    deepfakebench-ngc
```

### Issue 2: Path Resolution Issues
**Problem**: Relative paths not resolving correctly in container
**Solutions**:
1. **Working Directory**: Set correct working directory
2. **Path Joining**: Use `os.path.join()` for path construction
3. **Path Validation**: Validate all paths exist

**Path Resolution**:
```python
import os

def resolve_path(path, base_dir='/workspace'):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(base_dir, path)

def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path
```

### Issue 3: Symbolic Link Issues
**Problem**: Symbolic links not working in Docker container
**Solutions**:
1. **Copy Instead of Link**: Copy files instead of using symlinks
2. **Bind Mounts**: Use bind mounts for shared directories
3. **Volume Permissions**: Set correct volume permissions

---

## Network & Port Dependencies

### Issue 1: Port Conflicts
**Problem**: Ports already in use by other services
**Default Ports**:
- 8050 (Dashboard)
- 5432 (PostgreSQL)
- 6000 (TensorBoard)

**Solutions**:
1. **Port Mapping**: Map to different host ports
2. **Service Management**: Stop conflicting services
3. **Port Detection**: Detect available ports

**Port Management**:
```bash
# Check port usage
netstat -tulpn | grep :8050
netstat -tulpn | grep :5432

# Use different ports
docker run -p 8051:8050 -p 5433:5432 deepfakebench-ngc
```

### Issue 2: Network Connectivity
**Problem**: Container cannot access external services
**Solutions**:
1. **Network Mode**: Use appropriate network mode
2. **DNS Resolution**: Configure DNS resolution
3. **Firewall Rules**: Configure firewall rules

**Network Configuration**:
```bash
# Use host network for full access
docker run --network host deepfakebench-ngc

# Or use custom network
docker network create deepfake-network
docker run --network deepfake-network deepfakebench-ngc
```

### Issue 3: Service Dependencies
**Problem**: Services not starting in correct order
**Solutions**:
1. **Health Checks**: Implement health checks
2. **Service Dependencies**: Define service dependencies
3. **Startup Scripts**: Use startup scripts for service orchestration

**Service Orchestration**:
```bash
#!/bin/bash
# startup.sh

# Start PostgreSQL
service postgresql start

# Wait for PostgreSQL to be ready
while ! pg_isready -h localhost -p 5432; do
    sleep 1
done

# Start dashboard
python simple_dashboard.py &

# Start training/testing
exec "$@"
```

---

## Build & Compilation Issues

### Issue 1: C++ Compilation Errors
**Problem**: C++ extensions failing to compile
**Common Errors**:
- Missing C++ headers
- Incompatible compiler versions
- Missing build tools

**Solutions**:
```dockerfile
# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set compiler flags
ENV CC=gcc
ENV CXX=g++
ENV CFLAGS="-O2 -fPIC"
ENV CXXFLAGS="-O2 -fPIC"
```

### Issue 2: Python Extension Compilation
**Problem**: Python packages with C extensions failing to build
**Affected Packages**:
- `dlib`
- `psycopg2`
- `opencv-python`

**Solutions**:
```bash
# Install development headers
apt-get install -y \
    libpq-dev \
    libopencv-dev \
    libdlib-dev \
    python3-dev

# Install with specific flags
pip install --no-binary=dlib dlib==19.24.0
pip install --no-binary=psycopg2 psycopg2-binary
```

### Issue 3: CUDA Compilation Issues
**Problem**: CUDA extensions failing to compile
**Solutions**:
1. **CUDA Toolkit**: Install appropriate CUDA toolkit
2. **NVCC**: Ensure NVCC is available
3. **CUDA Libraries**: Install CUDA libraries

**CUDA Setup**:
```dockerfile
# Install CUDA toolkit
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-1 \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## Version Compatibility Matrix

### PyTorch & CUDA Compatibility
| PyTorch Version | CUDA Version | RTX 4090 Support | Notes |
|----------------|--------------|------------------|-------|
| 1.12.0 | 11.3 | ❌ | Too old, no RTX 4090 support |
| 2.0.0 | 11.8 | ⚠️ | Limited support |
| 2.0.1 | 12.1 | ✅ | Recommended |
| 2.1.0 | 12.1 | ✅ | Latest stable |

### Python Package Versions
| Package | Compatible Versions | Notes |
|---------|-------------------|-------|
| torch | 2.0.1+cu121 | CUDA 12.1 support |
| torchvision | 0.15.2+cu121 | Matches torch version |
| numpy | 1.24.3 | Stable version |
| opencv-python | 4.8.0.74 | Latest stable |
| scikit-learn | 1.3.0 | Latest stable |
| psycopg2-binary | 2.9.7 | PostgreSQL adapter |
| dash | 2.14.1 | Web dashboard |
| plotly | 5.17.0 | Interactive charts |

### System Requirements
| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | RTX 3080 | RTX 4090 | 24GB VRAM recommended |
| RAM | 32GB | 64GB | For large datasets |
| Storage | 500GB | 1TB | SSD recommended |
| CUDA | 12.1 | 12.2 | Latest driver |
| Docker | 20.10+ | 24.0+ | Latest version |

---

## Troubleshooting Guide

### Common Error Messages & Solutions

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`
**Solutions**:
```python
# Reduce batch size
batch_size = 8  # Instead of 32

# Clear GPU memory
torch.cuda.empty_cache()

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. Module Import Errors
**Error**: `ModuleNotFoundError: No module named 'training'`
**Solutions**:
```bash
# Set PYTHONPATH
export PYTHONPATH=/workspace:$PYTHONPATH

# Or in Python
import sys
sys.path.append('/workspace')
```

#### 3. Database Connection Errors
**Error**: `psycopg2.OperationalError: could not connect to server`
**Solutions**:
```bash
# Check PostgreSQL status
service postgresql status

# Check port availability
netstat -tulpn | grep :5432

# Test connection
psql -h localhost -p 5432 -U deepfake_user -d deepfake_bench
```

#### 4. File Not Found Errors
**Error**: `FileNotFoundError: [Errno 2] No such file or directory`
**Solutions**:
```python
# Verify file exists
import os
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    # Handle missing file

# Use absolute paths
absolute_path = os.path.abspath(relative_path)
```

#### 5. Memory Issues
**Error**: `Killed: 9` or `OSError: [Errno 28] No space left on device`
**Solutions**:
```bash
# Check memory usage
free -h
df -h

# Clear system cache
sync
echo 3 > /proc/sys/vm/drop_caches

# Increase shared memory
docker run --shm-size=64g ...
```

### Diagnostic Commands

#### System Diagnostics
```bash
# Check GPU status
nvidia-smi

# Check memory usage
free -h
df -h

# Check running processes
ps aux | grep python

# Check network connectivity
netstat -tulpn
```

#### Docker Diagnostics
```bash
# Check container status
docker ps -a

# Check container logs
docker logs deepfakebench-ngc

# Check container resources
docker stats deepfakebench-ngc

# Enter container for debugging
docker exec -it deepfakebench-ngc bash
```

#### Python Diagnostics
```python
# Check PyTorch installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Check package versions
import pkg_resources
packages = ['torch', 'torchvision', 'numpy', 'opencv-python', 'scikit-learn']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except:
        print(f"{package}: not installed")
```

### Recovery Procedures

#### 1. Complete System Reset
```bash
# Stop and remove container
docker stop deepfakebench-ngc
docker rm deepfakebench-ngc

# Remove image
docker rmi nvcr.io/nvidia/pytorch:23.08-py3

# Clear system cache
docker system prune -a

# Restart from scratch
./setup.sh
```

#### 2. Database Recovery
```bash
# Stop PostgreSQL
service postgresql stop

# Remove database files
rm -rf /var/lib/postgresql/data

# Recreate database
service postgresql start
createdb deepfake_bench
```

#### 3. Weight File Recovery
```bash
# Re-download weights
cd /workspace/training/weights
rm -rf *
wget https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/pretrained.zip
unzip pretrained.zip
```

This comprehensive dependency documentation covers all the technical challenges we faced during the DeepfakeBench migration, providing specific solutions and troubleshooting guidance for future deployments.
