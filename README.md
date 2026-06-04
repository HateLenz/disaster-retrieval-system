# Disaster Retrieval System

面向 xBD 灾害遥感数据的本地离线灾后检索系统。系统输入灾后建筑或灾后瓦片图像，在灾前图库中检索最相关的灾前影像，并提供模型训练、特征索引、FastAPI 后端、Vue/Electron 前端、检索历史、反馈标注、报告导出和索引维护能力。

> 说明：GitHub 仓库不应提交 xBD 原始数据、处理后的 patch、预训练模型、训练 checkpoint、FAISS/NPY 索引、实验输出和 SQLite 数据库。本文档会说明这些大文件应如何下载、生成和放置。

## 项目状态

当前代码已经跑通两条主线：

- **算法实验链路**：xBD 原始数据 -> tile 元数据 -> building patch -> Stage-1/Stage-2 训练 -> 特征提取 -> FAISS 检索评估 -> 案例和诊断报告。
- **本地应用链路**：训练好的模型 + 灾前特征库 -> FastAPI 检索服务 -> Vue/Electron 界面 -> 用户、历史、反馈、失败案例和报告导出。

默认稳定部署模型是 Stage-1 纯视觉检索模型 `clip_visual_baseline_smoke_stage1_v2.pt`。Stage-2 语义增强模型已经有训练脚本和配置，但目前更适合作为实验方向，不是默认应用模型。

## 目录结构

本文档所在目录就是项目根目录。GitHub 仓库建议直接以这一层作为根目录，不再额外套一层同名父目录。

```text
.
├── .gitignore
├── .python-version
├── README.md
├── environment.yml
├── image.ipynb                  # 可选的分析/展示 notebook
├── configs/                     # 训练和实验 profile
├── data/                        # 本地数据目录，不提交到 Git
├── docs/                        # 需求、设计、实现、测试和算法文档
├── frontend/                    # Vue + Electron 前端
├── pretrained/                  # 本地 CLIP 预训练权重，不提交权重文件
├── requirements/                # Python 依赖锁定文件
├── scripts/                     # 数据处理、训练、评估、索引构建脚本
└── src/                         # 后端、模型、数据集、检索工具源码
```

根目录 `.gitignore` 已经排除了以下大文件目录：

```text
data/raw/
data/interim/
data/processed/
pretrained/*
checkpoints/
indexes/
outputs/
db/*.db
frontend/node_modules/
frontend/dist/
```

## 外部资源清单

| 资源 | 是否必须 | 放置位置 | 获取方式 |
| --- | --- | --- | --- |
| xBD 原始数据 | 从头处理和训练必须 | `data/raw/xbd/geotiffs/` | 从 xBD/xView2 官方页面下载 |
| 处理后 building patch | 训练、评估、应用都需要 | `data/processed/` | 由脚本生成，或从你自己的外部存储下载 |
| CLIP 预训练权重 | 训练和推理必须 | `pretrained/openai-clip-vit-base-patch32/` | `scripts/download_clip_model.py` 或手动下载 Hugging Face snapshot |
| 项目 checkpoint | 应用推理必须 | `checkpoints/` | 自己训练生成，或从外部存储下载 |
| 应用检索索引 | 启动后端推荐准备 | `indexes/api_hold_stage1_v2/` | `scripts/build_api_index.py` 生成 |
| SQLite 数据库 | 应用运行生成 | `db/` | `retrieval_app.db` 首次启动自动创建，图库库由索引脚本生成 |

建议把这些资源放到网盘、对象存储、Hugging Face Dataset/Model、GitHub Release 或学校服务器中，并在你自己的 README 发布版本里补充下载链接。

## 环境准备

克隆仓库后先进入项目根目录；后续命令默认都在这个目录执行：

```powershell
cd disaster-retrieval-system
```

推荐使用 Python 3.10：

```powershell
conda create -n dfr-algo python=3.10.13 -y
conda activate dfr-algo
```

也可以直接使用仓库里的环境文件：

```powershell
conda env create -f environment.yml
conda activate dfr-algo
```

按你的 CUDA/CPU 环境安装 PyTorch 和 torchvision。建议参考 PyTorch 官方安装命令：

```powershell
# 示例：CPU 环境
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

再安装项目依赖：

```powershell
pip install -r requirements/algo.lock.txt
```

前端需要 Node.js 和 npm：

```powershell
node -v
npm -v
```

如果没有版本输出，先安装 Node.js LTS。

## 下载和组织 xBD 数据集

原始数据来自 xBD / xView2 Building Damage Assessment 数据集。请从官方页面获取数据：

- xBD / xView2 Dataset: <https://xview2.org/dataset>

下载后，把 `geotiffs` 数据整理成下面的结构：

```text
.
└── data/
    └── raw/
        └── xbd/
            └── geotiffs/
                ├── hold/
                │   ├── images/
                │   │   ├── <tile_id>_pre_disaster.tif
                │   │   └── <tile_id>_post_disaster.tif
                │   └── labels/
                │       ├── <tile_id>_pre_disaster.json
                │       └── <tile_id>_post_disaster.json
                ├── test/
                │   ├── images/
                │   └── labels/
                ├── tier1/
                │   ├── images/
                │   └── labels/
                └── tier3/
                    ├── images/
                    └── labels/
```

注意事项：

- 本项目当前只保留 `fire`、`flooding`、`wind` 三类主干灾害。
- 本地数据中使用的 split 是 `hold`、`test`、`tier1`、`tier3`，没有 `tier2`。
- 每个 tile 应有 4 个文件：灾前 TIF、灾后 TIF、灾前 JSON、灾后 JSON。
- 不建议用 `--absolute-paths` 生成 CSV。默认 repo-relative 路径更适合迁移项目。

可以用下面的命令检查目录：

```powershell
Get-ChildItem data/raw/xbd/geotiffs
Get-ChildItem data/raw/xbd/geotiffs/hold/images -Filter "*_pre_disaster.tif" | Measure-Object
Get-ChildItem data/raw/xbd/geotiffs/hold/labels -Filter "*_post_disaster.json" | Measure-Object
```

## 准备 CLIP 预训练权重

默认模型目录是：

```text
pretrained/openai-clip-vit-base-patch32/
```

联网环境可以运行：

```powershell
python scripts/download_clip_model.py
```

也可以手动从 Hugging Face 下载 `openai/clip-vit-base-patch32` 的完整 snapshot，并复制到上述目录：

- Hugging Face model: <https://huggingface.co/openai/clip-vit-base-patch32>

目录内至少应包含：

```text
config.json
preprocessor_config.json
pytorch_model.bin
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
```

训练、特征提取、评估和后端默认都使用 `local_files_only=true`，因此离线运行前必须准备好这个目录。

## 数据处理流程

### 1. 构建 tile 级 CSV

```powershell
python scripts/build_metadata.py `
  --data-root data/raw/xbd/geotiffs `
  --output data/processed/xbd_tile_dataset.csv
```

输出：

```text
data/processed/xbd_tile_dataset.csv
```

这个 CSV 每行对应一对灾前/灾后 tile，包含 `tile_id`、路径、`disaster`、`disaster_type`、`split`、建筑数量、损毁比例和文本提示等字段。

### 2. 构建 building 级 patch

```powershell
python scripts/build_building_patches.py `
  --tile-csv data/processed/xbd_tile_dataset.csv `
  --output-csv data/processed/xbd_building_patches.csv `
  --output-dir data/processed/xbd_building_patches `
  --patch-size 224 `
  --margin 16 `
  --stretch-mode percentile `
  --stretch-scope pair
```

输出：

```text
data/processed/xbd_building_patches.csv
data/processed/xbd_building_patches/
```

处理逻辑：

- 通过 xBD 标注中的 `uid` 对齐同一建筑的灾前/灾后实例。
- 使用灾前 bbox 和灾后 bbox 的并集裁剪。
- 额外外扩 `16px` 上下文。
- 输出统一 `224 x 224` 的 `_pre.png` 和 `_post.png`。
- 默认过滤 `un-classified`。

当前项目文档记录的完整 building patch 规模是 `286,791` 对，实际数量以你重新运行脚本后的 CSV 为准。

### 3. 构建 smoke 子集

```powershell
python scripts/build_smoke_subset.py `
  --csv data/processed/xbd_building_patches.csv `
  --output-csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --summary-json outputs/logs/smoke_stage1_visual/subset_summary.json `
  --group-by disaster_type damage_label `
  --split-limit tier3=4096 `
  --split-limit hold=1024 `
  --max-per-tile 24
```

输出：

```text
data/processed/xbd_building_patches_smoke_stage1.csv
outputs/logs/smoke_stage1_visual/subset_summary.json
```

smoke 子集不是随便抽样，而是按 `disaster_type + damage_label` 轮转采样，并限制单个 tile 的样本数。当前配置约 `5,120` 条，适合快速调试训练、评估和日志流程。

如果你只想运行应用，并且已经有 checkpoint 和应用索引，可以跳过原始 xBD 处理，只准备 `data/processed/xbd_building_patches.csv`、patch 图像和索引文件。

## 训练和实验

统一实验入口是：

```powershell
python scripts/run_retrieval_experiment.py --profile <profile.yaml>
```

它会按 profile 自动编排：

```text
subset -> train -> extract -> eval -> tile_eval -> cases -> sanity -> limitations
```

如果某个 profile 没有配置某一步，该步骤会自动跳过。查看实际命令但不执行：

```powershell
python scripts/run_retrieval_experiment.py `
  --profile configs/experiment/smoke_stage1_visual.yaml `
  --dry-run
```

### Stage-1 纯视觉训练

| 目标 | 配置文件 | 命令 |
| --- | --- | --- |
| smoke 基线 | `configs/experiment/smoke_stage1_visual.yaml` | `python scripts/run_retrieval_experiment.py --profile configs/experiment/smoke_stage1_visual.yaml` |
| smoke v2 | `configs/experiment/smoke_stage1_visual_v2.yaml` | `python scripts/run_retrieval_experiment.py --profile configs/experiment/smoke_stage1_visual_v2.yaml` |
| full 全量 | `configs/experiment/full_stage1_visual.yaml` | `python scripts/run_retrieval_experiment.py --profile configs/experiment/full_stage1_visual.yaml` |

Stage-1 模型结构：

```text
pre patch  -> CLIP Vision Encoder -> Projection Head -> L2 normalize
post patch -> CLIP Vision Encoder -> Projection Head -> L2 normalize
```

训练使用对称 InfoNCE，默认冻结 CLIP backbone，只训练 projection head。配置中默认启用灾害类型平衡采样和 `inverse_freq` loss weighting。

只训练模型、不跑后续评估时，可以直接调用脚本：

```powershell
python scripts/train_clip_visual_baseline.py `
  --csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --train-split tier3 `
  --eval-split hold `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --model-name pretrained/openai-clip-vit-base-patch32 `
  --batch-size 32 `
  --epochs 12 `
  --lr 0.0001 `
  --device cuda `
  --amp `
  --local-files-only `
  --loss-weighting inverse_freq `
  --selection-metric macro_recall@1
```

### Stage-2 语义增强训练

Stage-2 在查询侧融合灾害文本语义，训练脚本是：

```text
scripts/train_clip_semantic_stage2.py
```

已有配置：

```text
configs/smoke_stage2_semantic.yaml
configs/smoke_stage2_semantic_v2.yaml
configs/smoke_stage2_semantic_v3.yaml
configs/smoke_stage2_semantic_v4_stage1init.yaml
configs/smoke_stage2_semantic_v5_stage1init_type_damage.yaml
```

推荐优先跑 v5：

```powershell
python scripts/run_retrieval_experiment.py `
  --profile configs/smoke_stage2_semantic_v5_stage1init_type_damage.yaml
```

v4/v5 需要先有 Stage-1 初始化 checkpoint：

```text
checkpoints/clip_visual_baseline_smoke_stage1.pt
```

如果你想改为从 v2 初始化，修改对应 YAML 中的：

```yaml
train:
  init_visual_checkpoint: checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
```

单独运行 Stage-2 训练示例：

```powershell
python scripts/train_clip_semantic_stage2.py `
  --csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --train-split tier3 `
  --eval-split hold `
  --checkpoint checkpoints/clip_semantic_stage2_smoke_v5_stage1init_type_damage.pt `
  --init-visual-checkpoint checkpoints/clip_visual_baseline_smoke_stage1.pt `
  --model-name pretrained/openai-clip-vit-base-patch32 `
  --semantic-fields disaster_type,damage_label `
  --text-template "a post-disaster satellite image of a {damage_label} building after a {disaster_type} disaster" `
  --query-fusion-mode residual_gate `
  --detach-gallery-for-query-loss `
  --freeze-image-projection `
  --evaluate-before-training `
  --semantic-loss-weight 0.05 `
  --visual-loss-weight 0.0 `
  --device cuda `
  --amp `
  --local-files-only
```

### 特征提取和评估

提取灾前 gallery 特征：

```powershell
python scripts/extract_pre_clip_features.py `
  --csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --output indexes/pre_clip_features_smoke_stage1_v2.npy `
  --split hold `
  --view pre `
  --device cuda `
  --amp `
  --local-files-only
```

patch 级检索评估：

```powershell
python scripts/eval_clip_retrieval.py `
  --csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --features indexes/pre_clip_features_smoke_stage1_v2.npy `
  --index indexes/pre_clip_smoke_stage1_v2.faiss `
  --results-csv outputs/predictions/clip_retrieval_results_smoke_stage1_v2.csv `
  --metrics-json outputs/predictions/clip_retrieval_metrics_smoke_stage1_v2.json `
  --split hold `
  --top-k 10 `
  --device cuda `
  --amp `
  --local-files-only
```

tile 级聚合评估：

```powershell
python scripts/eval_clip_tile_retrieval.py `
  --csv data/processed/xbd_building_patches_smoke_stage1.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --features indexes/pre_clip_features_smoke_stage1_v2.npy `
  --index indexes/pre_clip_smoke_stage1_v2_tile.faiss `
  --results-csv outputs/predictions/clip_tile_retrieval_results_smoke_stage1_v2.csv `
  --metrics-json outputs/predictions/clip_tile_retrieval_metrics_smoke_stage1_v2.json `
  --split hold `
  --patch-top-k 100 `
  --tile-top-k 10 `
  --aggregation top_m `
  --top-m 5 `
  --device cuda `
  --amp `
  --local-files-only
```

可选分析脚本：

```powershell
# 已知 disaster 先验过滤实验
python scripts/eval_stage1v2_disaster_prior.py --prior-field disaster --amp --local-files-only

# 整图/瓦片滑窗检索实验
python scripts/eval_stage1v2_full_image_retrieval.py --prior-field disaster --amp --local-files-only

# Stage-1 smoke 超参数搜索
python scripts/search_stage1_smoke_hparams.py --dry-run
```

## 应用索引构建

后端应用不直接扫描原始 xBD，而是读取已经构建好的 hold 灾前特征库。构建命令：

```powershell
python scripts/build_api_index.py `
  --csv data/processed/xbd_building_patches.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --output-dir indexes/api_hold_stage1_v2 `
  --sqlite db/retrieval_hold_stage1_v2.db `
  --split hold `
  --device cuda `
  --amp `
  --local-files-only
```

输出：

```text
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
db/retrieval_hold_stage1_v2.db
```

后端加载优先级：

1. 如果存在 `indexes/api_hold_stage1_v2/pre_features.npy` 和 `pre_metadata.csv`，加载完整 hold 索引。
2. 否则尝试加载 smoke v2 索引：`indexes/pre_clip_features_smoke_stage1_v2.npy` 和对应 CSV。
3. 如果 checkpoint、features 或 metadata 缺失，`/api/health` 返回 `ready=false`，并提示缺失文件。

## 启动后端

在项目根目录执行：

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

检查状态：

```text
http://127.0.0.1:8000/api/health
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

首次启动会自动创建应用业务库：

```text
db/retrieval_app.db
```

并写入默认账号：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `admin` | `admin123` | 系统管理员 |
| `developer` | `developer123` | 算法开发人员 |
| `retriever` | `retriever123` | 检索用户 |

## 后端配置

后端配置位于 `src/api/retrieval_service.py`，可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DFR_CHECKPOINT_PATH` | `checkpoints/clip_visual_baseline_smoke_stage1_v2.pt` | 当前推理 checkpoint |
| `DFR_MODEL_NAME` | `pretrained/openai-clip-vit-base-patch32` | CLIP 权重目录或模型名 |
| `DFR_LOCAL_FILES_ONLY` | `true` | 是否只使用本地模型文件 |
| `DFR_FEATURES_PATH` | 自动选择完整 hold 或 smoke features | 自定义 `.npy` 特征矩阵 |
| `DFR_METADATA_PATH` | 自动选择完整 hold 或 smoke metadata | 自定义特征元信息 CSV |
| `DFR_SOURCE_CSV` | `data/processed/xbd_building_patches.csv` | 原始 building patch CSV |
| `DFR_SQLITE_PATH` | `db/retrieval_hold_stage1_v2.db` | 图库元信息 SQLite |
| `DFR_DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `DFR_IMAGE_SIZE` | `224` | CLIP 输入图像尺寸 |
| `DFR_BATCH_SIZE` | `64` | 推理 batch size |
| `DFR_AMP` | `true` | CUDA 下是否启用 AMP |
| `DFR_QUERY_WINDOW_SIZE` | `224` | 大图查询滑窗大小 |
| `DFR_QUERY_STRIDE` | `112` | 大图查询滑窗步长 |
| `DFR_MAX_QUERY_PATCHES` | `64` | 每张大图最多保留的 query patch 数 |
| `DFR_PATCH_TOP_K` | `100` | 每个 query patch 检索的候选 patch 数 |
| `DFR_AGGREGATION_TOP_M` | `20` | `top_m` 聚合使用的最高分数量 |

示例：

```powershell
$env:DFR_DEVICE="cpu"
$env:DFR_CHECKPOINT_PATH="checkpoints/clip_visual_baseline_smoke_stage1_v2.pt"
$env:DFR_FEATURES_PATH="indexes/api_hold_stage1_v2/pre_features.npy"
$env:DFR_METADATA_PATH="indexes/api_hold_stage1_v2/pre_metadata.csv"
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

## 启动 Web 前端

```powershell
cd frontend
npm install
npm run dev:web
```

访问：

```text
http://127.0.0.1:5173
```

前端默认请求：

```text
http://127.0.0.1:8000
```

如后端端口变化，设置：

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8001"
npm run dev:web
```

如果修改 Vite 端口或域名，也要同步检查 `src/api/app.py` 中的 CORS `allow_origins`。

## 启动 Electron 桌面端

Electron 默认会自动拉起后端：

```powershell
cd frontend
$env:DFR_BACKEND_CMD="python"
npm run dev:electron
```

如果需要指定 Conda 环境的 Python：

```powershell
$env:DFR_BACKEND_CMD="D:\Anaconda_envs\envs\dfr-algo\python.exe"
npm run dev:electron
```

如果后端已经手动启动，不希望 Electron 再启动后端：

```powershell
$env:DFR_ELECTRON_START_BACKEND="0"
npm run dev:electron
```

可配置的 Electron 环境变量：

| 环境变量 | 说明 |
| --- | --- |
| `DFR_ELECTRON_START_BACKEND=0` | 禁止 Electron 自动启动后端 |
| `DFR_BACKEND_CMD` | 后端命令，通常是 Python 解释器路径 |
| `DFR_BACKEND_ARGS` | 后端参数，默认 `-m uvicorn src.api.app:app --host 127.0.0.1 --port 8000` |
| `VITE_DEV_SERVER_URL` | Electron 开发模式加载的前端 URL，默认 `http://127.0.0.1:5173` |

构建前端静态产物：

```powershell
cd frontend
npm run build
```

构建后，如果 `frontend/dist` 存在，FastAPI 会把它挂载到 `/`，因此可直接通过后端地址访问应用。

## 界面功能与配置

前端核心文件：

```text
frontend/src/App.vue
frontend/src/styles.css
frontend/electron/main.cjs
```

默认界面配置点：

| 配置 | 位置 | 默认值 |
| --- | --- | --- |
| 后端地址 | `frontend/src/App.vue` 的 `API_BASE` | `VITE_API_BASE_URL` 或 `http://127.0.0.1:8000` |
| 界面 Top-K | `frontend/src/App.vue` 的 `SEARCH_TOP_K` | `10` |
| 过滤模式 | `filterMode` | `both` |
| 聚合策略 | `aggregation` | `top_m` |
| 允许切换的模型 | `src/api/retrieval_service.py` 的 `ALLOWED_APP_CHECKPOINTS` | `clip_visual_baseline_smoke_stage1.pt`、`clip_visual_baseline_smoke_stage1_v2.pt` |

前端主要页面：

- `检索`：上传单图、多图批量检索、样例检索、Top 10 灾前结果。
- `历史`：查看检索历史、恢复结果、导出 Markdown 报告。
- `失败案例`：低分和错误反馈样本复核。
- `系统状态`：模型、索引、设备、文件状态，支持索引重载和重建。
- `算法分析`：developer 查看策略统计、失败样本和模型切换。
- `用户管理`：admin 维护 retriever，developer 查看全部用户。

后端支持的过滤模式：

| 值 | 说明 |
| --- | --- |
| `both` | 同时按 `disaster_type` 和 `disaster` 过滤 |
| `type` | 只按 `disaster_type` 过滤 |
| `disaster` | 只按具体灾害事件过滤 |
| `none` | 不过滤，全库检索 |

后端支持的聚合策略：

| 值 | 说明 |
| --- | --- |
| `top_m` | 最高 m 个 patch 分数均值，默认策略 |
| `max` | 单个最高分 |
| `mean` | 命中分数均值 |
| `sum` | 命中分数求和 |
| `vote` | 命中次数投票 |

如果要让界面能切换新的 checkpoint：

1. 把 checkpoint 放到 `checkpoints/`。
2. 在 `src/api/retrieval_service.py` 的 `ALLOWED_APP_CHECKPOINTS` 中加入文件名和显示名。
3. 重启后端或在系统页点击“重载索引”。

## 主要 API

| 接口 | 权限 | 功能 |
| --- | --- | --- |
| `GET /api/health` | 公开 | 健康检查和模型/索引状态 |
| `POST /api/auth/login` | 公开 | 登录 |
| `POST /api/auth/register` | 公开 | 注册 |
| `GET /api/options` | 登录 | 灾害类型、事件和候选统计 |
| `GET /api/samples` | 登录 | hold 样例 |
| `POST /api/search` | 登录 | 上传单图检索 |
| `POST /api/search-by-path` | 登录 | 本地路径检索 |
| `POST /api/batch-search` | 登录 | 多图上传检索 |
| `GET /api/history` | 登录 | 检索历史 |
| `GET /api/history/{id}/report.md` | 登录 | 导出单条 Markdown 报告 |
| `GET /api/history/export` | 登录 | 批量导出历史报告包 |
| `POST /api/feedback` | 登录 | 结果反馈 |
| `GET /api/failures` | admin/developer | 失败案例 |
| `GET /api/system/status` | admin/developer | 系统状态 |
| `POST /api/index/reload` | admin/developer | 重载模型和索引 |
| `POST /api/index/rebuild` | admin/developer | 后台重建应用索引 |
| `GET /api/algorithm/analysis` | developer | 算法统计分析 |
| `GET /api/models` | developer | 可切换模型列表 |
| `POST /api/models/switch` | developer | 切换 checkpoint |
| `GET /api/users` | admin/developer | 用户列表 |

## 最小可运行清单

只做应用演示时，至少准备：

```text
pretrained/openai-clip-vit-base-patch32/
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
data/processed/xbd_building_patches.csv
data/processed/xbd_building_patches/
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
```

`db/retrieval_app.db` 可以由后端首次启动自动创建。`db/retrieval_hold_stage1_v2.db` 可以由 `scripts/build_api_index.py` 生成。

## 验证命令

Python 语法检查：

```powershell
python -m py_compile `
  src/api/app.py `
  src/api/retrieval_service.py `
  src/api/app_store.py `
  scripts/build_api_index.py `
  scripts/run_retrieval_experiment.py
```

后端健康检查：

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

另开终端：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

登录并访问选项：

```powershell
$login = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/auth/login `
  -ContentType application/json `
  -Body '{"username":"admin","password":"admin123"}'

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/options `
  -Headers @{Authorization="Bearer $($login.token)"}
```

## 常见问题

### `/api/health` 返回 `ready=false`

检查缺失项：

```text
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
pretrained/openai-clip-vit-base-patch32/
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
```

如果索引缺失，运行：

```powershell
python scripts/build_api_index.py --split hold --amp --local-files-only
```

### 训练时报找不到 CLIP 权重

确认：

```text
pretrained/openai-clip-vit-base-patch32/config.json
pretrained/openai-clip-vit-base-patch32/pytorch_model.bin
```

或者临时允许联网下载，把训练命令中的 `--local-files-only` 去掉。

### 前端无法连接后端

确认后端启动在：

```text
http://127.0.0.1:8000/api/health
```

如果后端端口不是 8000，启动前端前设置：

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:<port>"
```

### Electron 启动后后端未就绪

Electron 默认调用 `python`。如果系统默认 Python 不是项目环境，设置：

```powershell
$env:DFR_BACKEND_CMD="D:\Anaconda_envs\envs\dfr-algo\python.exe"
```

### 图片资源打不开

后端只允许访问项目目录内的资源，并会解析 `data/processed/`、`data/raw/`、`outputs/`、`indexes/` 等路径标记。不要把查询图像或结果图像放在项目外路径中。

### full 实验显存不足

先改 YAML：

```yaml
train:
  batch_size: 8
  num_workers: 2
  amp: true
```

或者使用 CPU：

```yaml
train:
  device: cpu
  amp: false
```

CPU 能跑通流程，但训练会明显变慢。

## 文档导航

更多细节见：

| 文档 | 内容 |
| --- | --- |
| `dataset.md` | 数据集来源、字段、统计和处理策略 |
| `pretrained/README.md` | 本地预训练模型目录说明 |
| `docs/algorithm_design.md` | Stage-1/Stage-2 算法、指标和实验结果 |
| `docs/application_project_overview.md` | 应用工程总览 |
| `docs/system_dev.md` | 本地启动和数据导入流程 |
| `docs/system_design.md` | 架构、数据库、模块和算法设计 |
| `docs/system_implementation.md` | 后端、前端和应用功能实现 |
| `docs/system_testing.md` | 测试范围、用例和验证记录 |
| `docs/frontend_code_by_module.md` | 前端模块和 API 对应关系 |

## 项目一句话

本项目已经形成“xBD 数据处理 -> CLIP 跨时相建筑检索训练 -> FAISS 灾前图库索引 -> FastAPI 检索服务 -> Vue/Electron 离线应用”的完整闭环；GitHub 仓库只保存代码和说明，大规模数据、权重和索引通过外部资源补齐。
