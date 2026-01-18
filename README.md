# 金融术语标准化系统（RAG）

本项目参考 `rag-project02-medical-nlp-box-master` 的实现方式，使用 **本地 FAISS 向量索引（`.faiss`）+ 向量检索**，基于 `万条金融标准术语.csv` 实现金融领域“专有名词标准化”。


## 本期学习：索引优化（Indexing）

参考：`rag-in-action/06-索引优化-Indexing`


### 01-从小块到大上下文（small-to-big）

- **索引什么**：检索仍在“小块”上进行（这里的小块=单条术语 term）。
- **返回什么**：命中后返回其对应的“大上下文”（这里用“相邻术语窗口”拼出更大的上下文）。
- **实现位置**：
  - 构建：`backend/tools_build_index_optimized.py`（输出 `fin_terms_small.*` + `fin_terms_big.*`）
  - 检索：`FinStdServiceOptimized` 的 `small_to_big` 模式

### 02-构建有层次的索引（hierarchical）

- **父索引**：父节点是“粗粒度块”，这里用每 N 条术语组成一个 parent block，并用前若干个词条拼出 summary。
- **子索引**：子节点是具体 term，带 parent_id。
- **检索流程**：先检索 parent，再在候选 parent 范围内筛 child（最小改动：child 仍用单一 FAISS，过滤在 Python 侧完成）。
- **实现位置**：
  - 构建：`backend/tools_build_index_optimized.py`（输出 `fin_terms_parent.*` + `fin_terms_child.*`）
  - 检索：`FinStdServiceOptimized` 的 `hierarchical` 模式

### 03-构建多表示的索引（multi-representation）

- **思路**：同一个 term 构造多种文本表示（去标点、缩写/首字母、term+label 等），统一指向 canonical term。
- **实现位置**：
  - 构建：`backend/tools_build_index_optimized.py`（输出 `fin_terms_multi.*`）
  - 检索：`FinStdServiceOptimized` 的 `multi` 模式（按 canonical term 去重）

## 构建“优化索引”

在 `backend/` 目录运行：

```bash
python tools_build_index_optimized.py
```

生成文件位于 `backend/db/`：
- `fin_terms_small.faiss` / `fin_terms_small_meta.jsonl`
- `fin_terms_big.faiss` / `fin_terms_big_meta.jsonl`
- `fin_terms_parent.faiss` / `fin_terms_parent_meta.jsonl`
- `fin_terms_child.faiss` / `fin_terms_child_meta.jsonl`
- `fin_terms_multi.faiss` / `fin_terms_multi_meta.jsonl`

## 调用（API）

新增接口：`POST /api/fin/std/optimized`

- `indexMode`：`baseline` / `small_to_big` / `hierarchical` / `multi`
- 其他参数与 `/api/fin/std` 基本相同

## 数据准备
## 前端（frontend）

### 目录结构

- `frontend/package.json`：依赖与启动脚本
- `frontend/src/App.js`：主路由与页面入口
- `frontend/src/pages/StdPage.js`：标准化页面（单条/批量查询）
- `frontend/src/components/Sidebar.js`：侧边栏导航
- `frontend/src/components/shared/ModelOptions.js`：embedding 配置表单
- `frontend/src/lib/api.js`：API 地址与请求封装
- `frontend/.env.example`：API 地址环境变量示例

### 安装依赖

```powershell
Set-Location -Path 'D:\Work\fin-term-std\frontend'
npm install
```

### 启动开发服务器

```powershell
npm start
```
默认浏览器访问 http://localhost:3000

### 生产构建

```powershell
npm run build
# 可用 serve -s build 启动静态服务
```

### API 地址配置

如需自定义后端地址，复制 `.env.example` 为 `.env` 并修改：

```
REACT_APP_API_BASE_URL=http://127.0.0.1:8009
```

## 前后端联调流程

1. 启动后端（见上文 backend 部分）
2. 启动前端（见上文 frontend 部分）
3. 浏览器访问 http://localhost:3000，进入“金融术语标准化”页面，输入术语即可调用后端接口，返回 TopK 标准术语候选。

## 常见问题

- **后端无法访问 huggingface.co，embedding 加载失败**：请参考 Troubleshooting 部分，或提前下载模型到本地并配置本地路径。
- **前端 Failed to fetch**：多为后端未启动、端口不对、API 地址配置不一致或防火墙拦截。
- **uvicorn 启动即退出**：请确保在 backend 目录下运行，命令参数无误。

---

把 `万条金融标准术语.csv` 放到：

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