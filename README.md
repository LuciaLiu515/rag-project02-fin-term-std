# 金融术语标准化系统（RAG）

本项目参考 `rag-project02-medical-nlp-box-master` 的实现方式，使用 **本地 FAISS 向量索引（`.faiss`）+ 向量检索**，基于 `万条金融标准术语.csv` 实现金融领域“专有名词标准化”。


## 把 `万条金融标准术语.csv` 放到：

- `backend/data/万条金融标准术语.csv`

CSV 预期格式：

- 第 1 列：标准术语（term）
- 第 2 列：标签（label，示例里为 `FINTERM`）


## 后端（backend）

### 1) 创建虚拟环境并安装依赖

```powershell
Set-Location -Path 'D:\Work\fin-term-std'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) 构建向量索引

```powershell
Set-Location -Path 'D:\Work\fin-term-std\backend'
D:\Work\fin-term-std\.venv\Scripts\python.exe .\tools_build_index.py
```

### 3) 启动后端

```powershell
Set-Location -Path 'D:\Work\fin-term-std\backend'
D:\Work\fin-term-std\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8009 --log-level debug
```

访问：`http://127.0.0.1:8009/docs`


## 前端（frontend）

### 1) 目录结构

- `frontend/package.json`：依赖与启动脚本
- `frontend/src/App.js`：主路由与页面入口
- `frontend/src/pages/StdPage.js`：标准化页面（单条/批量查询）
- `frontend/src/components/Sidebar.js`：侧边栏导航
- `frontend/src/components/shared/ModelOptions.js`：embedding 配置表单
- `frontend/src/lib/api.js`：API 地址与请求封装
- `frontend/.env.example`：API 地址环境变量示例

### 2) 安装依赖

```powershell
Set-Location -Path 'D:\Work\fin-term-std\frontend'
npm install
```

### 3) 启动开发服务器

```powershell
npm start
```
默认浏览器访问 http://localhost:3000
