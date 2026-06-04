# 灾后图像检索系统开发与启动说明

本文档说明系统的本地开发、数据导入、索引构建和完整启动流程。系统采用本地离线部署方式：后端加载本地 PyTorch 模型和本地 FAISS/NumPy 特征索引，前端通过 HTTP 调用本机 FastAPI 服务。

## 1. 项目目录

项目根目录为：

```text
D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system
```

主要目录说明如下：

```text
data/
  raw/xbd/geotiffs/                  原始 xBD 数据导入目录
  processed/                         处理后的瓦片元数据、建筑物 patch 清单和 patch 图像

checkpoints/
  clip_visual_baseline_smoke_stage1_v2.pt
                                      已训练好的 Stage-1 纯图像检索模型

pretrained/
  openai-clip-vit-base-patch32/       本地 CLIP 预训练权重目录

indexes/
  api_hold_stage1_v2/                 应用检索用 hold 灾前图像向量索引

db/
  retrieval_app.db                    应用业务数据库，启动后自动创建
  retrieval_hold_stage1_v2.db         hold 灾前检索库元数据数据库

src/api/
  app.py                              FastAPI 应用入口
  retrieval_service.py                检索模型、索引和图像资源服务
  app_store.py                        用户、历史、反馈等业务数据存储

frontend/
  src/                                Vue 前端页面
  electron/                           Electron 桌面端入口

scripts/
  build_metadata.py                   原始 xBD 元数据导入
  build_building_patches.py           建筑物 patch 数据生成
  build_smoke_subset.py               smoke 小样本数据生成
  build_api_index.py                  应用检索索引构建
```

## 2. 数据导入位置

### 2.1 原始 xBD 数据导入

原始 xBD 数据需要放在：

```text
data/raw/xbd/geotiffs/
```

推荐目录结构如下：

```text
data/raw/xbd/geotiffs/
  hold/
    images/
      <tile_id>_pre_disaster.tif
      <tile_id>_post_disaster.tif
    labels/
      <tile_id>_pre_disaster.json
      <tile_id>_post_disaster.json
  tier1/
    images/
    labels/
  tier3/
    images/
    labels/
  test/
    images/
    labels/
```

其中：

- `hold` 用作本系统后端灾前检索数据库。
- 用户上传的查询图像来自 `hold` 中的灾后图像。
- `tier1`、`tier3` 等训练集用于模型训练和实验。
- 文件命名需要保留 xBD 原始格式，例如 `xxx_pre_disaster.tif` 和 `xxx_post_disaster.tif`。

### 2.2 处理后数据位置

执行数据处理脚本后，系统会生成以下文件：

```text
data/processed/xbd_tile_dataset.csv
data/processed/xbd_building_patches.csv
data/processed/xbd_building_patches/
data/processed/xbd_building_patches_smoke_stage1.csv
```

含义如下：

- `xbd_tile_dataset.csv`：瓦片级元数据，记录灾害类型、灾害地区、split、灾前/灾后图像路径和标签路径。
- `xbd_building_patches.csv`：建筑物级 patch 元数据，记录每个建筑物对应的灾前 patch、灾后 patch、灾害类型、灾害地区和损毁等级。
- `xbd_building_patches/`：从原始大图中裁剪得到的建筑物 patch 图像。
- `xbd_building_patches_smoke_stage1.csv`：用于快速实验的 smoke 子集。

### 2.3 应用检索库位置

应用运行时不直接扫描原始 xBD 目录，而是读取已经构建好的检索索引：

```text
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
db/retrieval_hold_stage1_v2.db
```

其中：

- `pre_features.npy` 保存 hold 灾前图像特征向量。
- `pre_metadata.csv` 保存每个向量对应的灾前图像路径、灾害类型、灾害地区等元信息。
- `pre_features.faiss` 是 FAISS 检索索引。
- `retrieval_hold_stage1_v2.db` 是检索库元数据 SQLite 数据库。

因此，“数据导入”分为三步：

1. 把原始 xBD 数据放入 `data/raw/xbd/geotiffs/`。
2. 运行数据处理脚本，生成 `data/processed/` 下的瓦片和建筑物 patch 数据。
3. 运行索引构建脚本，把 hold 灾前数据导入应用检索库 `indexes/api_hold_stage1_v2/`。

当前系统的数据导入是离线脚本流程，不是在前端页面中上传原始 xBD 数据。前端的上传功能用于提交查询图像，不用于导入检索库。

## 3. 环境准备

### 3.1 Python 环境

推荐使用已经配置好的 Conda 环境：

```powershell
conda activate dfr-algo
```

如果 PowerShell 中默认 `python` 不是该环境，可以直接使用完整路径：

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" --version
```

后端主要依赖包括：

- `fastapi`
- `uvicorn`
- `torch`
- `numpy`
- `pandas`
- `pillow`
- `faiss-cpu` 或 `faiss-gpu`
- `transformers`
- `rasterio`
- `shapely`

### 3.2 前端环境

前端需要安装 Node.js 和 npm。检查命令：

```powershell
node -v
npm -v
```

如果没有输出版本号，需要先安装 Node.js。开发阶段建议使用 Node.js LTS 版本。

### 3.3 模型文件

确认最佳模型文件存在：

```text
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
```

确认本地 CLIP 预训练权重目录存在：

```text
pretrained/openai-clip-vit-base-patch32/
```

该系统理论上支持离线搜索，前提是模型文件、预训练权重、处理后数据和检索索引均已存在于本地。

## 4. 首次数据导入流程

以下命令均在项目根目录执行：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system
```

### 4.1 生成瓦片级元数据

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" scripts/build_metadata.py
```

该步骤读取：

```text
data/raw/xbd/geotiffs/
```

生成：

```text
data/processed/xbd_tile_dataset.csv
```

### 4.2 生成建筑物 patch 数据

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" scripts/build_building_patches.py
```

该步骤读取：

```text
data/processed/xbd_tile_dataset.csv
```

生成：

```text
data/processed/xbd_building_patches.csv
data/processed/xbd_building_patches/
```

建筑物 patch 是系统训练、评估和应用检索的基本图像单位。

### 4.3 生成 smoke 实验子集

如果需要复现实验或快速验证模型训练流程，可以生成 smoke 子集：

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" scripts/build_smoke_subset.py `
  --csv data/processed/xbd_building_patches.csv `
  --output-csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --summary-json outputs/logs/smoke_stage1_visual/subset_summary.json `
  --split-limit tier3=4096 `
  --split-limit hold=1024
```

如果只做应用启动，并且模型和索引已经存在，可以跳过该步骤。

## 5. 应用检索索引构建

后端检索服务需要 hold split 的灾前图像特征索引。使用最佳模型构建应用索引：

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" scripts/build_api_index.py `
  --csv data/processed/xbd_building_patches.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --split hold `
  --amp
```

构建完成后应生成：

```text
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
db/retrieval_hold_stage1_v2.db
```

如果没有完整 hold 索引，后端会尝试回退到 smoke 索引文件，但正式系统演示和论文测试建议使用完整 hold 索引。

## 6. 业务数据库初始化

应用业务数据库为：

```text
db/retrieval_app.db
```

该数据库在后端第一次启动时自动创建，包含：

- 用户表
- 会话表
- 检索历史表
- 检索结果表
- 反馈记录表

系统会自动初始化演示账号：

| 角色 | 用户名 | 密码 | 权限说明 |
| --- | --- | --- | --- |
| 管理员 | `admin` | `admin123` | 检索用户查询与增删改查、失败案例查看与 Markdown 导出包、索引管理、系统状态、检索、历史、反馈 |
| 算法开发人员 | `developer` | `developer123` | 查看全部用户、失败案例 Markdown 导出包、系统状态、算法分析、模型切换、索引调试、检索、历史、反馈 |
| 检索员 | `retriever` | `retriever123` | 单图检索、历史查看、反馈提交 |

也可以通过前端注册新用户。注册界面提供检索用户、系统管理员和算法开发人员三类角色下拉选择。

## 7. 完整启动应用

### 7.1 启动后端服务

在项目根目录执行：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system

& "D:\Anaconda_envs\envs\dfr-algo\python.exe" -m uvicorn src.api.app:app `
  --host 127.0.0.1 `
  --port 8000
```

启动成功后访问：

```text
http://127.0.0.1:8000/api/health
```

如果返回 `ready: true`，表示模型、索引和数据库均已加载完成。

也可以访问接口文档：

```text
http://127.0.0.1:8000/docs
```

### 7.2 验证后端登录接口

可以用 PowerShell 简单验证：

```powershell
$login = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/auth/login `
  -ContentType application/json `
  -Body '{"username":"admin","password":"admin123"}'

$token = $login.token

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/options `
  -Headers @{Authorization="Bearer $token"}
```

如果能返回灾害类型、灾害地区和检索参数选项，说明后端服务可用。

### 7.3 启动 Web 前端

打开新的 PowerShell 窗口：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system\frontend

npm install
npm run dev:web
```

启动成功后访问：

```text
http://127.0.0.1:5173
```

Web 前端默认调用：

```text
http://127.0.0.1:8000
```

因此需要先启动后端，再启动前端。

### 7.4 启动 Electron 桌面端

如果需要桌面应用形式运行：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system\frontend

npm install
$env:DFR_BACKEND_CMD="D:\Anaconda_envs\envs\dfr-algo\python.exe"
npm run dev:electron
```

如果后端已经手动启动，不希望 Electron 再启动一个后端进程，可以执行：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system\frontend

$env:DFR_ELECTRON_START_BACKEND="0"
npm run dev:electron
```

## 8. 日常开发启动顺序

在模型、索引和数据库已经准备好的情况下，日常开发只需要执行以下步骤。

第一步，启动后端：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system

& "D:\Anaconda_envs\envs\dfr-algo\python.exe" -m uvicorn src.api.app:app `
  --host 127.0.0.1 `
  --port 8000
```

第二步，启动前端：

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system\frontend

npm run dev:web
```

第三步，浏览器打开：

```text
http://127.0.0.1:5173
```

第四步，登录系统：

```text
用户名：admin
密码：admin123
```

或：

```text
用户名：developer
密码：developer123
```

或：

```text
用户名：retriever
密码：retriever123
```

## 9. 前端功能入口

登录后系统按角色展示不同功能。

### 9.1 检索工作台

主要功能：

- 上传 hold 灾后图像。
- 选择 `disaster_type`。
- 选择 `disaster`。
- 设置 Top K、过滤方式和聚合方式。
- 获取 Top 5 灾前候选图像。
- 查看灾前/灾后图像对比。
- 提交结果反馈。

### 9.2 检索历史

主要功能：

- 查看当前用户或全局检索历史。
- 按灾害类型、灾害地区和时间追踪检索记录。
- 重新打开历史检索详情。
- 导出 Markdown 报告压缩包，压缩包内包含 `report.md` 和对应图像。

### 9.3 失败案例库

主要功能：

- 汇总低置信度或用户标记不满意的检索记录。
- 点击失败案例后弹出复核窗口，查看灾后查询图像和灾前候选图像。
- 为后续模型优化和消融实验提供样本依据。

### 9.4 系统状态与索引管理

主要功能：

- 查看后端模型、索引、数据库和运行设备状态。
- 重新加载索引。
- 重建应用索引。
- 查看当前候选灾害类型和灾害地区。

### 9.5 用户管理

管理员可见，主要功能：

- 查看注册用户列表。
- 管理员不显示 developer 用户信息，只对检索用户进行新增、修改、删除和按用户名查询。
- 用户管理界面仅展示用户名、密码重置入口和角色类别，不再单独展示显示名称字段。
- 算法开发人员可以查看全部用户信息，但不负责账号修改。
- 区分系统管理员、算法开发人员和检索用户界面权限。

## 10. 常见问题

### 10.1 后端返回 ready=false

优先检查以下文件是否存在：

```text
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
```

如果索引不存在，重新执行：

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" scripts/build_api_index.py `
  --csv data/processed/xbd_building_patches.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --split hold `
  --amp
```

### 10.2 前端无法连接后端

检查后端是否启动：

```text
http://127.0.0.1:8000/api/health
```

如果 8000 端口被占用，可以换端口启动后端，但需要同步修改前端 API 地址。

### 10.3 Electron 启动失败

常见原因是 Electron 默认调用的 `python` 不是 `dfr-algo` 环境。设置：

```powershell
$env:DFR_BACKEND_CMD="D:\Anaconda_envs\envs\dfr-algo\python.exe"
npm run dev:electron
```

### 10.4 npm 命令不可用

说明本机没有安装 Node.js，或 Node.js 没有加入系统 `PATH`。安装 Node.js 后重新打开 PowerShell，再执行：

```powershell
node -v
npm -v
```

### 10.5 查询图像无法显示

系统只允许读取项目目录内的数据和资源文件。确认图像路径位于：

```text
data/
indexes/
outputs/
```

或通过前端上传查询图像，而不是输入项目外部路径。

## 11. 最小可运行清单

在演示系统前，至少确认以下内容存在：

```text
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
pretrained/openai-clip-vit-base-patch32/
data/processed/xbd_building_patches.csv
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
db/retrieval_app.db
```

如果 `db/retrieval_app.db` 不存在，启动后端会自动创建。其他模型、数据和索引文件需要提前通过离线脚本生成。
