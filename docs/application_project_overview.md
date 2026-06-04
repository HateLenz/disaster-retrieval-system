# 灾后检索系统应用工程项目梳理

## 1. 项目定位

本项目是一个面向灾后遥感影像检索场景的本地离线应用系统。系统以 xBD 数据集中的灾前/灾后建筑级影像为基础，使用本地 CLIP 视觉检索模型和 FAISS 向量索引，实现“输入灾后图像，检索 Top 5 灾前图像”的业务流程。

项目不是单一算法脚本，而是一个完整应用工程，覆盖数据处理、模型训练与评估、索引构建、后端 API、前端交互、桌面端封装、用户权限、检索历史、反馈标注、报告导出和系统维护。

核心目标如下：

| 类型 | 目标 |
| --- | --- |
| 算法目标 | 对灾后图像编码，在灾前图库中完成向量检索和 Top-K 排序 |
| 应用目标 | 提供可交互、可演示、可离线运行、可维护的桌面检索系统 |
| 工程目标 | 将模型、索引、数据、接口、前端和本地数据库组织成稳定闭环 |
| 论文目标 | 支撑需求分析、系统设计、系统实现、系统测试和实验分析章节 |

## 2. 业务闭环

系统围绕灾情研判中的灾前灾后影像对比需求设计。典型业务闭环为：

```text
用户登录
  -> 选择灾害类型和灾害地区
  -> 输入灾后图像
  -> 执行检索
  -> 返回 Top 5 灾前图像
  -> 查看前后对比和元信息
  -> 提交反馈
  -> 保存历史
  -> 导出报告
```

业务价值主要体现在：

- 使用本地灾前数据库减少人工查找历史影像的成本。
- 使用 `disaster_type` 和 `disaster` 作为业务先验，缩小候选范围。
- 通过 Top 5 结果、相似度分数和元信息辅助人工判断。
- 通过历史、反馈和失败案例支撑复盘与模型优化。
- 支持离线部署，适合无公网或网络不稳定环境演示。

## 3. 用户角色与权限

系统设置三类角色，不再单独设置“研判人员”角色。

| 角色 | 定位 | 主要功能 |
| --- | --- | --- |
| `retriever` 检索用户 | 业务使用者 | 图像检索、结果查看、本人历史、反馈、报告导出 |
| `admin` 系统管理员 | 运行维护者 | 用户管理、系统状态、索引重载与重建、全部历史、失败案例 |
| `developer` 算法开发人员 | 模型分析者 | 全部历史、失败案例、算法分析、模型切换、全部用户查看 |

权限控制要点：

- 未登录用户不能访问检索、历史、管理类接口。
- 检索用户只能查看本人历史。
- 管理员可维护检索用户，但不查看 developer 用户信息。
- 算法开发人员可查看全部用户和算法分析信息，但不负责账号修改。
- 索引管理允许 `admin` 和 `developer`。
- 模型切换和算法分析仅允许 `developer`。

## 4. 总体架构

系统采用单机本地离线部署架构，由四层组成。

```text
Electron + Vue 前端
        |
        | HTTP / JSON / multipart
        v
FastAPI 应用服务
        |
        +-- 用户认证、权限控制、历史、反馈、报告
        |
        v
PyTorch + CLIP + FAISS 检索服务
        |
        v
SQLite / CSV / NPY / FAISS / 本地图像文件
```

各层职责如下：

| 层次 | 技术 | 职责 |
| --- | --- | --- |
| 桌面端交互层 | Electron + Vue 3 + Vite | 登录注册、角色化工作台、图像上传、结果展示、历史和报告 |
| 应用服务层 | FastAPI + SQLite | REST API、鉴权、权限控制、业务数据持久化、资源访问 |
| 检索算法层 | PyTorch + CLIP + FAISS | 模型加载、图像编码、候选过滤、向量检索、结果聚合 |
| 本地数据层 | SQLite + CSV + NPY + FAISS + xBD 图像 | 用户、会话、检索历史、图库元信息、特征矩阵和索引 |

## 5. 工程目录

主要目录和职责如下：

| 路径 | 说明 |
| --- | --- |
| `docs/` | 业务分析、需求分析、系统设计、实现、测试和工程图文档 |
| `data/raw/xbd/geotiffs/` | xBD 原始灾前/灾后影像与标注 |
| `data/processed/` | 瓦片级元数据、建筑级 patch 清单和 patch 图像 |
| `checkpoints/` | 已训练模型 checkpoint |
| `pretrained/openai-clip-vit-base-patch32/` | 本地 CLIP 预训练权重 |
| `indexes/` | 灾前特征矩阵、FAISS 索引和元信息 CSV |
| `db/` | 应用业务数据库和图库元信息数据库 |
| `src/api/` | FastAPI 入口、业务存储和检索服务封装 |
| `src/datasets/` | xBD tile 数据集和 building patch 数据集读取 |
| `src/models/` | CLIP 视觉编码器和 checkpoint 加载 |
| `src/retrieval/` | FAISS/NumPy 检索、指标和后处理 |
| `src/engine/` | 训练和评估通用逻辑 |
| `scripts/` | 数据处理、训练、评估、特征提取和索引构建脚本 |
| `frontend/` | Vue 前端和 Electron 桌面端 |
| `outputs/` | 日志、导出报告、上传缓存和实验输出 |

## 6. 数据工程

数据处理采用“两级组织”：

1. 瓦片级元数据。
2. 建筑级配对 patch。

### 6.1 原始数据

原始 xBD 数据位于：

```text
data/raw/xbd/geotiffs/
```

按 `hold`、`test`、`tier1`、`tier3` 组织，每个 tile 包含：

- 灾前影像：`*_pre_disaster.tif`
- 灾后影像：`*_post_disaster.tif`
- 灾前标注：`*_pre_disaster.json`
- 灾后标注：`*_post_disaster.json`

项目主要保留三类灾害：

- `fire`
- `flooding`
- `wind`

### 6.2 瓦片级数据

生成脚本：

```text
scripts/build_metadata.py
```

输出：

```text
data/processed/xbd_tile_dataset.csv
```

核心字段包括：

- `tile_id`
- `pre_image_path`
- `post_image_path`
- `pre_label_path`
- `post_label_path`
- `disaster`
- `disaster_type`
- `split`
- 建筑数量和损毁统计
- `damage_ratio`
- `severe_damage_ratio`
- `text_prompt`

### 6.3 建筑级 patch 数据

生成脚本：

```text
scripts/build_building_patches.py
```

输出：

```text
data/processed/xbd_building_patches.csv
data/processed/xbd_building_patches/
```

处理逻辑：

1. 通过 `uid` 对齐灾前和灾后同一建筑。
2. 解析建筑多边形和边界框。
3. 使用灾前框和灾后框并集作为裁剪区域。
4. 外扩 16 像素上下文。
5. 裁剪灾前/灾后 patch。
6. 统一缩放为 `224 x 224`。
7. 默认过滤 `un-classified` 标签。

建筑级 patch 是训练、评估和应用检索的基本单位。

### 6.4 应用检索库

应用运行不直接扫描原始 xBD 目录，而是读取已经构建好的 hold 灾前检索库：

```text
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
db/retrieval_hold_stage1_v2.db
```

其中：

- `pre_features.npy` 保存灾前 patch 特征矩阵。
- `pre_metadata.csv` 保存与特征行号对齐的元信息。
- `pre_features.faiss` 保存 FAISS 内积索引。
- `retrieval_hold_stage1_v2.db` 保存图库元信息。

## 7. 算法链路

算法任务是建筑级跨时相图像检索：

```text
Query: 灾后建筑图像 q_post
Gallery: 灾前建筑图像集合 G_pre
Output: Top-K 灾前建筑图像列表
```

### 7.1 Stage-1 视觉检索基线

当前默认部署模型：

```text
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
```

模型结构：

```text
灾前 patch -> CLIP Vision Encoder -> Projection Head -> L2 normalize
灾后 patch -> CLIP Vision Encoder -> Projection Head -> L2 normalize
```

关键设计：

- 使用本地 `openai-clip-vit-base-patch32` 权重。
- 冻结 CLIP vision backbone。
- 增加 projection head 输出 256 维检索向量。
- 使用 L2 归一化。
- 使用对称 InfoNCE 损失训练跨时相对齐。
- 使用 FAISS `IndexFlatIP` 检索，归一化后内积等价于余弦相似度。

### 7.2 Stage-2 语义增强方向

Stage-2 在 Stage-1 基础上引入文本语义：

- `disaster_type`
- `damage_label`

文本模板：

```text
a post-disaster satellite image of a {damage_label} building after a {disaster_type} disaster
```

当前文档结论是：Stage-2 是最终优化方向，但默认应用部署仍使用 Stage-1 visual v2 作为稳定模型。原因是直接引入语义可能破坏已学习的视觉检索空间，需要以 Stage-1 初始化和轻量语义约束稳定训练。

### 7.3 候选过滤和聚合

应用侧支持候选过滤：

| 模式 | 说明 |
| --- | --- |
| `both` | 同时按灾害类型和灾害地区过滤 |
| `type` | 仅按灾害类型过滤 |
| `disaster` | 仅按灾害地区过滤 |
| `none` | 全库检索 |

应用侧支持聚合策略：

| 策略 | 说明 |
| --- | --- |
| `top_m` | 最高 m 个 patch 分数均值，默认策略 |
| `max` | 单个最高分 |
| `mean` | 命中分数均值 |
| `sum` | 命中分数求和 |
| `vote` | 命中次数投票 |

文档中的实验结论显示，使用 `disaster` 先验过滤能明显提升检索结果，因此业务输入和算法检索是互补关系。

## 8. 后端服务

后端入口：

```text
src/api/app.py
```

核心服务：

```text
src/api/retrieval_service.py
src/api/app_store.py
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `app.py` | FastAPI 应用、接口路由、鉴权依赖、报告导出、索引任务 |
| `retrieval_service.py` | 模型加载、索引加载、路径解析、图像编码、检索和聚合 |
| `app_store.py` | SQLite 用户、会话、历史、结果、反馈、失败案例和分析统计 |

主要接口如下：

| 接口 | 功能 |
| --- | --- |
| `GET /api/health` | 健康检查和检索服务状态 |
| `POST /api/auth/register` | 注册本地用户 |
| `POST /api/auth/login` | 登录并获取 token |
| `GET /api/auth/me` | 根据 token 恢复会话 |
| `POST /api/auth/logout` | 退出登录 |
| `GET /api/options` | 获取灾害类型、地区、过滤和聚合选项 |
| `GET /api/samples` | 获取 hold 样例图像 |
| `POST /api/search` | 上传单图检索 |
| `POST /api/search-by-path` | 使用本地路径检索 |
| `POST /api/batch-search` | 批量上传检索 |
| `POST /api/batch-search-by-path` | 批量路径检索 |
| `GET /api/history` | 获取检索历史 |
| `GET /api/history/{search_id}` | 获取历史详情 |
| `GET /api/history/{search_id}/report` | 导出 Markdown 报告 zip |
| `GET /api/history/export` | 批量导出历史报告包 |
| `GET /api/results/{result_id}` | 获取单个结果详情 |
| `POST /api/feedback` | 提交结果反馈 |
| `GET /api/failures` | 获取失败案例 |
| `GET /api/failures/export` | 导出失败案例报告包 |
| `GET /api/system/status` | 系统状态 |
| `GET /api/index/status` | 索引任务状态 |
| `POST /api/index/reload` | 重载模型和索引 |
| `POST /api/index/rebuild` | 后台重建索引 |
| `GET /api/algorithm/analysis` | 算法统计分析 |
| `GET /api/models` | 查看本地模型列表 |
| `POST /api/models/switch` | 切换模型 checkpoint |
| `GET /api/users` | 查看用户 |
| `POST /api/users` | 新增用户 |
| `PATCH /api/users/{user_id}` | 修改用户 |
| `DELETE /api/users/{user_id}` | 删除或停用用户 |
| `GET /api/assets` | 安全访问项目内图像资源 |

## 9. 数据库设计

系统使用两类 SQLite 数据库。

### 9.1 应用业务库

路径：

```text
db/retrieval_app.db
```

主要表：

| 表 | 说明 |
| --- | --- |
| `users` | 本地用户账号、密码摘要、角色 |
| `sessions` | 本地会话 token |
| `search_records` | 一次检索的查询条件、耗时、Top 分数和索引范围 |
| `search_results` | 每个 Top-K 结果的排名、分数、路径和元信息 |
| `feedback_records` | 用户对结果的 correct/wrong/uncertain 反馈 |

### 9.2 图库元信息库

路径：

```text
db/retrieval_hold_stage1_v2.db
```

主要表：

| 表 | 说明 |
| --- | --- |
| `gallery_items` | 灾前图库 patch 的路径、灾害类型、事件、建筑 ID、损毁标签等 |

业务库记录系统使用过程，图库库记录检索对象。两者解耦，便于重建索引或切换模型。

## 10. 前端与桌面端

前端目录：

```text
frontend/
```

核心文件：

| 文件 | 说明 |
| --- | --- |
| `frontend/src/App.vue` | Vue 单页主工作台 |
| `frontend/src/main.js` | Vue 入口 |
| `frontend/src/styles.css` | 页面样式 |
| `frontend/electron/main.cjs` | Electron 主进程 |
| `frontend/package.json` | 前端依赖和启动脚本 |

前端脚本：

```text
npm run dev:web       # 启动 Web 前端
npm run dev:electron  # 启动 Electron 桌面端
npm run build         # 构建前端产物
npm run preview       # 预览构建产物
```

前端功能入口：

| 页面 | 功能 |
| --- | --- |
| 登录注册 | 本地账号登录、注册和会话恢复 |
| 检索工作台 | 单图检索、批量检索、样例检索、Top 5 展示 |
| 历史 | 历史列表、详情恢复、报告导出 |
| 失败案例 | 低分和错误反馈样本复核、导出 |
| 系统状态 | 模型、索引、设备、文件状态和索引任务 |
| 用户管理 | 管理员维护 retriever，开发人员查看全部用户 |
| 算法分析 | developer 查看策略统计、反馈统计和模型切换 |

## 11. 离线运行资源

最小可运行资源清单：

```text
checkpoints/clip_visual_baseline_smoke_stage1_v2.pt
pretrained/openai-clip-vit-base-patch32/
data/processed/xbd_building_patches.csv
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
db/retrieval_app.db
```

其中 `db/retrieval_app.db` 可由后端首次启动自动创建；模型、CLIP 权重、处理后数据和索引需要提前准备。

默认演示账号：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `admin` | `admin123` | 系统管理员 |
| `developer` | `developer123` | 算法开发人员 |
| `retriever` | `retriever123` | 检索用户 |

## 12. 启动流程

### 12.1 后端

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system

& "D:\Anaconda_envs\envs\dfr-algo\python.exe" -m uvicorn src.api.app:app `
  --host 127.0.0.1 `
  --port 8000
```

健康检查：

```text
http://127.0.0.1:8000/api/health
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

### 12.2 Web 前端

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system\frontend
npm run dev:web
```

访问：

```text
http://127.0.0.1:5173
```

### 12.3 Electron 桌面端

```powershell
cd D:\products\bishe\disaster-retrieval-system\disaster-retrieval-system\frontend
npm run dev:electron
```

如果后端已手动启动，可设置：

```powershell
$env:DFR_ELECTRON_START_BACKEND="0"
npm run dev:electron
```

## 13. 索引构建流程

完整 hold 灾前索引构建命令：

```powershell
& "D:\Anaconda_envs\envs\dfr-algo\python.exe" scripts/build_api_index.py `
  --csv data/processed/xbd_building_patches.csv `
  --checkpoint checkpoints/clip_visual_baseline_smoke_stage1_v2.pt `
  --split hold `
  --amp
```

构建产物：

```text
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
db/retrieval_hold_stage1_v2.db
```

如果完整索引不存在，后端会尝试回退到 smoke 索引，但正式演示建议使用完整 hold 索引。

## 14. 测试与验收

测试目标覆盖：

- 后端能加载本地模型、索引和元信息。
- API 能正常完成鉴权、检索、历史、反馈和报告。
- 前端能完成上传、样例选择、检索和结果展示。
- 系统能在模型或索引缺失时返回明确错误。
- 资源接口能限制项目外路径访问。
- 角色权限符合设计。

关键验收标准：

| 编号 | 标准 |
| --- | --- |
| 1 | 本地能启动后端和前端 |
| 2 | `/api/health` 返回服务状态 |
| 3 | 能加载本地 checkpoint 和检索索引 |
| 4 | 能选择 `disaster_type` 和 `disaster` |
| 5 | 能上传或选择灾后图像并返回 Top 5 |
| 6 | 能查看结果详情和前后对比数据 |
| 7 | 能保存检索历史和用户反馈 |
| 8 | 能导出 Markdown 报告包 |
| 9 | 管理员能查看系统状态并维护索引 |
| 10 | developer 能查看算法分析和切换模型 |

当前文档记录的验证状态：

- 后端核心链路已完成接口级验证。
- 登录、检索、反馈、历史、报告、系统状态等接口通过。
- 当前环境曾记录缺少 Node/npm，因此 Vue/Electron 端到端 UI 测试需要在前端环境具备后继续验证。

## 15. 工程亮点

1. 算法链路完整：从 xBD 原始数据处理到 building patch、训练、评估、特征提取和索引构建。
2. 应用闭环完整：登录、检索、结果、反馈、历史、报告和失败案例形成业务闭环。
3. 离线能力明确：模型、权重、索引、数据库和图像资源均可本地部署。
4. 角色权限清晰：检索用户、管理员、算法开发人员职责分离。
5. 检索策略可配置：支持候选过滤模式和多种聚合策略。
6. 运维能力具备：支持系统状态、索引重载、索引重建和模型切换。
7. 论文结构支撑充分：需求、设计、实现、测试、图表和数据处理文档齐全。

## 16. 风险与改进点

| 风险或不足 | 影响 | 建议 |
| --- | --- | --- |
| 前端端到端测试依赖 Node/npm 环境 | UI 验证可能不完整 | 在目标演示机安装 Node.js 并补充截图或测试记录 |
| 完整 hold 索引构建耗时与硬件相关 | 演示前可能无法临时重建 | 提前生成并备份 `indexes/api_hold_stage1_v2/` |
| Stage-2 语义增强效果尚未超过 Stage-1 稳定基线 | 默认模型仍是视觉基线 | 论文中明确 Stage-2 是后续优化方向 |
| xBD 数据规模大 | 迁移和部署成本高 | 演示机保留必要 hold 数据和索引，训练数据可按需保留 |
| Electron 自动拉起后端依赖 Python 环境路径 | 桌面端可能启动失败 | 设置 `DFR_BACKEND_CMD` 指向 `dfr-algo` 环境 Python |
| 资源访问依赖项目内路径规范 | 历史路径迁移后可能失效 | 继续保留路径解析兼容逻辑，并避免项目外图像路径 |

## 17. 一句话总结

该项目是一个以 xBD 建筑级灾前/灾后影像为数据基础、以 CLIP 视觉检索和 FAISS 索引为算法核心、以 FastAPI + Vue/Electron + SQLite 为应用框架的本地离线灾后遥感图像检索系统，已经具备从数据处理、模型部署到多角色应用演示和测试验收的完整工程形态。
