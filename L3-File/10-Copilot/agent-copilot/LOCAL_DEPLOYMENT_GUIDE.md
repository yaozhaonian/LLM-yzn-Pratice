# 本地部署指南

本文档详细说明如何在本地环境中搭建数据库并运行该项目。

## 一、环境要求

- Python 3.8+
- Docker & Docker Compose (用于部署 Milvus)
- MongoDB 5.0+

## 二、MongoDB 安装与配置

### 2.1 安装 MongoDB

#### macOS (使用 Homebrew)
```bash
brew tap mongodb/brew
brew install mongodb-community@7.0
```

#### Ubuntu/Debian
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
```

### 2.2 启动 MongoDB

```bash
# macOS
brew services start mongodb-community@7.0

# Ubuntu/Debian
sudo systemctl start mongod
sudo systemctl enable mongod
```

### 2.3 创建管理员用户

```bash
# 连接到 MongoDB
mongo

# 切换到 admin 数据库
use admin

# 创建管理员用户
db.createUser({
    user: "admin",
    pwd: "123456",
    roles: [{ role: "root", db: "admin" }]
})

# 创建项目数据库和用户
use tools
db.createUser({
    user: "admin",
    pwd: "123456",
    roles: [{ role: "readWrite", db: "tools" }]
})

# 验证用户创建
db.auth("admin", "123456")
```

### 2.4 启用认证 (可选但推荐)

编辑 MongoDB 配置文件：

```bash
# macOS: /usr/local/etc/mongod.conf
# Ubuntu/Debian: /etc/mongod.conf

# 添加或修改以下配置
security:
    authorization: enabled
```

重启 MongoDB：

```bash
# macOS
brew services restart mongodb-community@7.0

# Ubuntu/Debian
sudo systemctl restart mongod
```

## 三、Milvus 向量数据库安装与配置

### 3.1 使用 Docker Compose 启动 Milvus

创建 `milvus-docker-compose.yml` 文件：

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls=http://0.0.0.0:2379 --data-dir=/etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.1
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3

networks:
  default:
    name: milvus
```

启动 Milvus：

```bash
mkdir -p volumes/etcd volumes/minio volumes/milvus
docker-compose -f milvus-docker-compose.yml up -d
```

### 3.2 验证 Milvus 运行状态

```bash
docker-compose -f milvus-docker-compose.yml ps
```

确保所有服务状态为 `Up`。

## 四、Ollama 大语言模型安装

### 4.1 安装 Ollama

```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh
```

### 4.2 启动 Ollama 服务

```bash
ollama serve
```

### 4.3 拉取模型

```bash
# 拉取 Qwen2.5 7B 模型
ollama pull qwen2.5:7b

# 拉取 BGE-M3 嵌入模型
ollama pull bge-m3:latest
```

## 五、项目配置与运行

### 5.1 修改环境变量

编辑 `.env` 文件，根据实际情况修改配置：

```bash
# MongoDB 配置
mongo_host=127.0.0.1
mongo_port=27017
mongo_db=tools
mongo_user=admin
mongo_password=123456
auth_source=admin

# Milvus 配置
milvus_uri=http://localhost:19530
milvus_db_name=tool_db

# 模型配置
model_name=qwen2.5:7b
model_temperature=0.01
model_top_p=0.01

# API 配置
sim_api_key=hihachengfeng
```

### 5.2 安装 Python 依赖

```bash
cd agent-copilot
pip install -r requirements.txt
```

### 5.3 启动项目

```bash
python app.py
```

项目将在 `http://localhost:5005` 启动。

## 六、工具数据初始化

### 6.1 通过 API 上传工具定义

使用 curl 上传 OpenAPI 规范文件：

```bash
curl -X POST http://localhost:5005/upload_file \
  -F "file=@api_data/dataset_apis_aliyun.json"
```

### 6.2 验证工具是否已加载

```bash
curl http://localhost:5005/get_all_tools
```

### 6.3 测试 API 规划功能

```bash
curl -X POST http://localhost:5005/api_planning \
  -H "Content-Type: application/json" \
  -d '{
    "query": "查询产品名为苹果的产品信息",
    "modelName": "qwen2.5:7b",
    "temperature": 0.01,
    "api_key": "your_api_key",
    "api_url": "http://127.0.0.1:11434",
    "isCopilot": true,
    "isContext": false,
    "contexts": [],
    "contextNumber": 10
  }'
```

## 七、使用自定义后端 API

如果您想使用自己的后端 API 而不是默认的 `http://121.43.198.13:8080`，需要：

### 7.1 修改 API 定义文件

编辑 `api_data/dataset_apis_aliyun.json`，修改 `servers` 部分：

```json
"servers": [
    {
        "url": "http://your-backend-api:port",
        "description": "Your custom backend API"
    }
]
```

### 7.2 确保 API 兼容 OpenAPI 3.0 规范

您的后端 API 需要遵循以下格式：

- 使用 OpenAPI 3.0 规范定义
- 支持 JSON 请求/响应
- 包含 `operationId`、`summary`、`description` 等字段
- 参数定义包含 `name`、`in`、`description`、`schema` 等字段

### 7.3 更新 API 密钥

修改 `.env` 文件中的 `sim_api_key`：

```bash
sim_api_key=your_custom_api_key
```

## 八、常见问题

### Q1: MongoDB 连接失败

**原因**：可能是 MongoDB 未启动或认证配置不正确。

**解决**：
```bash
# 检查 MongoDB 状态
brew services list | grep mongodb

# 查看日志
tail -f /usr/local/var/log/mongodb/mongo.log
```

### Q2: Milvus 连接失败

**原因**：Milvus 容器未启动或网络不通。

**解决**：
```bash
# 检查容器状态
docker-compose -f milvus-docker-compose.yml ps

# 查看 Milvus 日志
docker logs milvus-standalone
```

### Q3: 模型调用失败

**原因**：Ollama 服务未启动或模型未下载。

**解决**：
```bash
# 检查 Ollama 服务
ps aux | grep ollama

# 检查模型是否存在
ollama list
```

### Q4: 工具调用返回错误

**原因**：后端 API 不可用或 API 密钥不正确。

**解决**：
```bash
# 测试后端 API
curl http://121.43.198.13:8080/products/getProductByName \
  -H "Content-Type: application/json" \
  -H "X-API_Key: hihachengfeng" \
  -d '{"name": "苹果"}'
```

## 九、数据备份与恢复

### 9.1 MongoDB 备份

```bash
mongodump --uri="mongodb://admin:123456@127.0.0.1:27017/tools" --out=backup/
```

### 9.2 MongoDB 恢复

```bash
mongorestore --uri="mongodb://admin:123456@127.0.0.1:27017/tools" backup/tools/
```

### 9.3 Milvus 备份

Milvus 数据存储在 `volumes/milvus` 目录，直接备份该目录即可。
