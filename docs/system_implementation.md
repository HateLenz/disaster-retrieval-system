# 灾后检索系统实现

## 1. 实现概述

系统基于已有的 Stage-1 视觉检索算法实现，将训练得到的 `clip_visual_baseline_smoke_stage1_v2.pt` 封装为本地检索服务，并通过 Electron + Vue 提供桌面端交互界面。

当前实现覆盖以下能力：

- 后端加载本地 checkpoint 和本地 CLIP 预训练权重。
- 后端加载 hold 灾前特征库和元信息。
- 支持按 `disaster_type` 和 `disaster` 过滤候选库。
- 支持上传灾后图像检索 Top 5 灾前图像。
- 支持从 hold 样例中选择灾后图像检索。
- 支持通过 HTTP 接口返回图像资源。
- 支持构建完整 hold 灾前索引。
- 前端展示系统状态、筛选项、查询图像、Top 5 结果和相似度分数。

## 2. 代码结构

```text
disaster-retrieval-system/
├── src/
│   ├── api/
│   │   ├── app.py                  # FastAPI 入口
│   │   └── retrieval_service.py    # 检索服务封装
│   ├── datasets/
│   │   ├── transforms.py           # CLIP 图像预处理
│   │   └── xbd_building_dataset.py # xBD building patch 数据集
│   ├── models/
│   │   └── clip_visual_encoder.py  # CLIP 视觉编码器与 checkpoint 加载
│   └── retrieval/
│       └── faiss_index.py          # FAISS / NumPy 检索封装
├── scripts/
│   └── build_api_index.py          # 后端 hold 灾前库索引构建脚本
├── frontend/
│   ├── electron/main.cjs           # Electron 主进程
│   ├── src/App.vue                 # Vue 主界面
│   ├── src/main.js                 # Vue 入口
│   └── src/styles.css              # 前端样式
└── docs/
    ├── requirements_analysis.md
    ├── system_design.md
    ├── system_implementation.md
    └── system_testing.md
```

## 3. 后端实现

### 3.1 FastAPI 入口

文件：`src/api/app.py`

后端通过 FastAPI 暴露检索接口。应用启动时创建 `DisasterRetrievalService` 实例，并在 startup 阶段加载模型、索引和元信息。

主要接口如下：

| 接口 | 功能 |
| --- | --- |
| `/api/health` | 返回系统状态 |
| `/api/options` | 返回可选灾害类型和灾害地区 |
| `/api/samples` | 返回 hold 样例图像 |
| `/api/search` | 上传图像检索 |
| `/api/search-by-path` | 使用已有路径检索 |
| `/api/assets` | 返回本地图像文件 |

接口错误通过 HTTP 状态码反馈：

- `400`：请求参数或检索条件错误。
- `403`：请求访问仓库外文件。
- `404`：图像资源不存在。
- `503`：模型或索引尚未就绪。
- `500`：未预期的服务错误。

### 3.2 检索服务封装

文件：`src/api/retrieval_service.py`

`DisasterRetrievalService` 是后端核心类，主要职责包括：

1. 读取环境变量和默认路径。
2. 加载本地 checkpoint。
3. 加载灾前特征矩阵和元信息表。
4. 维护候选过滤后的 FAISS 索引缓存。
5. 对查询图像进行预处理和编码。
6. 执行向量检索并聚合结果。
7. 格式化返回给前端的 JSON。

默认路径如下：

| 资源 | 默认路径 |
| --- | --- |
| checkpoint | `checkpoints/clip_visual_baseline_smoke_stage1_v2.pt` |
| 完整 hold 特征 | `indexes/api_hold_stage1_v2/pre_features.npy` |
| 完整 hold 元信息 | `indexes/api_hold_stage1_v2/pre_metadata.csv` |
| smoke hold 特征 | `indexes/pre_clip_features_smoke_stage1_v2.npy` |
| smoke hold 元信息 | `indexes/pre_clip_features_smoke_stage1_v2.csv` |
| SQLite | `db/retrieval_hold_stage1_v2.db` |

当完整 hold 索引存在时，系统优先加载完整索引；否则退回到已有 smoke_stage1_v2 hold 索引，保证开发和演示流程可运行。

### 3.3 图像编码

后端复用已有 CLIP 图像预处理：

```python
build_clip_transform(image_size=224, train=False)
```

查询图像分两种情况处理：

- 建筑级 patch：直接 resize / center crop 后编码。
- 较大图像：使用滑窗切分为多个 patch，分别编码后聚合检索结果。

编码输出为归一化向量，和灾前库向量使用余弦相似度进行匹配。

### 3.4 候选过滤与索引缓存

用户输入 `disaster_type` 和 `disaster` 后，系统先在元信息表中过滤候选，再对候选特征构建或读取缓存索引。

缓存键为：

```text
(disaster_type, disaster)
```

这样相同灾害类型和地区的重复查询无需重复构建索引，提高交互效率。

### 3.5 结果聚合

当查询图像被切成多个 patch 时，每个 query patch 会返回多个灾前候选 patch。系统按 `tile_id` 聚合分数：

```text
tile_score = mean(top_m_patch_scores)
```

返回结果包含：

- rank
- score
- best_patch_score
- patch_hit_count
- tile_id
- positive_id
- building_uid
- disaster
- disaster_type
- damage_label
- pre_patch_url
- pre_image_url
- tile_patch_count

## 4. 索引构建实现

文件：`scripts/build_api_index.py`

该脚本用于构建后端实际使用的 hold 灾前数据库。

执行流程：

1. 读取 `data/processed/xbd_building_patches.csv`。
2. 筛选 `split=hold`。
3. 加载 `clip_visual_baseline_smoke_stage1_v2.pt`。
4. 批量读取灾前 patch。
5. 提取灾前特征并保存为 `.npy`。
6. 保存元信息 CSV。
7. 构建 FAISS 索引。
8. 写入 SQLite 元信息表。

推荐命令：

```bash
cd disaster-retrieval-system
python scripts/build_api_index.py --split hold --amp
```

输出：

```text
indexes/api_hold_stage1_v2/pre_features.npy
indexes/api_hold_stage1_v2/pre_metadata.csv
indexes/api_hold_stage1_v2/pre_features.faiss
db/retrieval_hold_stage1_v2.db
```

## 5. 前端实现

### 5.1 Electron 主进程

文件：`frontend/electron/main.cjs`

Electron 启动后会创建桌面窗口，并默认尝试启动本地 FastAPI 服务：

```text
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

如果后端已经手动启动，可以通过环境变量关闭自动启动：

```bash
DFR_ELECTRON_START_BACKEND=0
```

### 5.2 Vue 主界面

文件：`frontend/src/App.vue`

页面分为左右两栏：

- 左侧：灾害类型选择、灾害地区选择、图像上传、查询预览、hold 样例列表。
- 右侧：系统状态、候选库信息、Top 5 检索结果。

前端启动时调用：

1. `/api/health` 获取系统状态。
2. `/api/options` 获取筛选项。
3. `/api/samples` 获取样例图像。

执行检索时：

- 上传文件使用 `/api/search`。
- 样例路径使用 `/api/search-by-path`。

### 5.3 前端视觉设计

界面设计强调实用性：

- 不做营销式首页，启动后直接进入检索工作区。
- 使用状态条显示模型和索引是否可用。
- 结果卡片显示图像、排名、分数和元信息。
- 样例列表用于快速演示系统能力。
- 颜色采用浅色工作台风格，适合数据检索类系统。

## 6. 当前已实现功能清单

| 功能 | 实现位置 |
| --- | --- |
| 后端启动与服务状态 | `src/api/app.py` |
| 模型和索引加载 | `src/api/retrieval_service.py` |
| Top 5 检索 | `src/api/retrieval_service.py` |
| 图像上传检索 | `POST /api/search` |
| 路径样例检索 | `POST /api/search-by-path` |
| 图片资源访问 | `GET /api/assets` |
| hold 样例列表 | `GET /api/samples` |
| 完整 hold 索引构建 | `scripts/build_api_index.py` |
| Electron 桌面入口 | `frontend/electron/main.cjs` |
| Vue 检索界面 | `frontend/src/App.vue` |

## 7. 应用业务功能实现概览

当前应用已经实现检索历史、结果详情、前后对比、报告导出、用户反馈、失败案例库、批量检索、系统状态、索引管理、用户管理、算法分析和模型切换。上述功能与需求分析中的五个业务用例保持一致，并由 `src/api/app.py`、`src/api/app_store.py`、`src/api/retrieval_service.py` 和 `frontend/src/App.vue` 共同实现。

每次检索完成后，后端会写入 `search_records` 和 `search_results` 表，保存查询图像、检索条件、Top 5 结果、检索耗时、模型和索引范围。前端可在历史页面恢复一次检索记录，并可导出单条 Markdown 报告或批量 Markdown 报告包。

结果详情面板展示查询灾后图像、命中灾前图像、相似度、tile、building 和损毁标签等元信息，并提供“正确 / 错误 / 不确定”反馈入口。反馈记录写入 `feedback_records` 表，并用于失败案例库和算法分析统计。

系统状态页展示模型 checkpoint、推理设备、图库规模、索引文件、SQLite 文件和后台索引任务状态。管理员和算法开发人员可以触发索引重载和完整 hold 索引重建；算法开发人员还可以查看本地 checkpoint 列表并切换当前检索模型。

## 8. 运行方式

后端：

```bash
cd disaster-retrieval-system
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

前端 Web 调试：

```bash
cd disaster-retrieval-system/frontend
npm install
npm run dev:web
```

Electron 调试：

```bash
cd disaster-retrieval-system/frontend
npm run dev:electron
```

## 9. 环境变量

| 变量 | 说明 |
| --- | --- |
| `DFR_CHECKPOINT_PATH` | 指定 checkpoint 路径 |
| `DFR_FEATURES_PATH` | 指定特征矩阵路径 |
| `DFR_METADATA_PATH` | 指定元信息 CSV 路径 |
| `DFR_SOURCE_CSV` | 指定源数据 CSV 路径 |
| `DFR_SQLITE_PATH` | 指定 SQLite 路径 |
| `DFR_DEVICE` | 指定 `cuda` 或 `cpu` |
| `DFR_LOCAL_FILES_ONLY` | 是否仅使用本地模型文件 |
| `DFR_QUERY_WINDOW_SIZE` | 查询图像滑窗大小 |
| `DFR_QUERY_STRIDE` | 查询图像滑窗步长 |
| `DFR_PATCH_TOP_K` | patch 级候选数 |
| `DFR_AGGREGATION_TOP_M` | tile 聚合时使用的 top-m 数量 |

## 10. 实现小结

当前系统已经从纯算法脚本扩展为可运行的本地检索应用。后端完成模型、索引、数据库和 API 的封装；前端完成基本检索交互和结果展示；索引构建脚本支持完整 hold 灾前库接入。

为了让应用部分更符合毕业设计系统要求，后续应优先补充检索历史、结果详情、前后对比、报告导出和索引管理。这些功能不改变核心算法，但能明显增强系统的业务完整性和工程展示价值。

## 11. 应用增强功能实现

根据需求分析和系统设计，应用侧已进一步补充以下功能。

### 11.1 多用户角色与登录注册

后端新增 `src/api/app_store.py`，使用本地 SQLite 数据库 `db/retrieval_app.db` 保存用户、会话、检索历史、结果和反馈。

支持角色：

| 角色 | 说明 | 界面权限 |
| --- | --- | --- |
| `retriever` | 检索用户 | 检索、查看本人历史、反馈和报告导出 |
| `admin` | 系统管理员 | 检索、全部历史、失败案例查看和导出、系统状态、索引维护、retriever 用户查询与增删改查 |
| `developer` | 算法开发人员 | 检索、全部历史、失败案例导出、系统状态、索引维护、算法分析、模型切换、全部用户查看 |

系统首次启动时会创建演示账号：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `admin` | `admin123` | 系统管理员 |
| `developer` | `developer123` | 算法开发人员 |
| `retriever` | `retriever123` | 检索用户 |

### 11.2 检索历史与详情

每次上传检索、样例路径检索或批量检索完成后，后端会自动写入：

- `search_records`：一次检索的查询条件、用户、耗时、模型索引范围和结果数量。
- `search_results`：每个 Top-K 结果的排名、分数、图像路径和完整 JSON。

前端提供“历史”页面。检索用户只能查看自己的历史；管理员和算法开发人员可查看全部历史。

### 11.3 前后对比详情

检索结果卡片支持点击进入详情。详情面板展示：

- 查询灾后图像。
- 命中灾前图像。
- tile、building、score、damage 等元信息。
- 用户反馈入口。

### 11.4 报告导出

新增接口：

```http
GET /api/history/{search_id}/report
```

系统将一次检索记录导出为 Markdown 报告压缩包，包含 `report.md`、查询图像以及对应的灾前候选图像，报告中记录查询条件、用户、耗时、索引范围和 Top 结果表。

### 11.5 用户反馈与失败案例库

新增反馈接口：

```http
POST /api/feedback
```

用户可对检索结果标记：

- `correct`
- `wrong`
- `uncertain`

失败案例库会聚合低分检索记录和被标记为 `wrong` 的结果，供系统管理员查看和算法开发人员分析。

### 11.6 批量检索

新增接口：

```http
POST /api/batch-search
POST /api/batch-search-by-path
```

前端上传框支持多选图像。批量检索会逐张执行检索，保存每张图像的历史记录，并在界面上显示每个文件的完成或失败状态。

### 11.7 筛选和排序策略

检索接口新增参数：

| 参数 | 可选值 | 说明 |
| --- | --- | --- |
| `filter_mode` | `both`、`type`、`disaster`、`none` | 控制候选库过滤方式 |
| `aggregation` | `top_m`、`max`、`mean`、`sum`、`vote` | 控制多 patch 结果聚合策略 |

这使系统可以在演示时对比不同业务约束和排序策略对结果的影响。

### 11.8 系统状态与索引管理

新增接口：

```http
GET /api/system/status
GET /api/index/status
POST /api/index/reload
POST /api/index/rebuild
```

系统状态页展示：

- 推理设备。
- PyTorch / CUDA 状态。
- checkpoint 路径。
- 特征和元信息文件是否存在。
- 当前索引范围和图库规模。
- 后台索引构建任务状态。

管理员和算法开发人员可以在界面中触发索引重载和完整 hold 索引重建。

## 12. 界面功能实现

系统前端采用 Vue 3 实现单页角色化工作台，主要代码位于：

```text
frontend/src/App.vue
frontend/src/styles.css
```

前端通过 `fetch` 调用本地 FastAPI 接口，后端地址默认为：

```text
http://127.0.0.1:8000
```

登录成功后，前端将后端返回的 token 保存到 `localStorage` 中，之后每次请求都在请求头中携带：

```text
Authorization: Bearer <token>
```

为保证论文结构一致，界面实现按照需求分析中的五个用例进行归并。注册登录是系统入口，因此在实现章节中单独展开；其余界面功能均归入对应业务模块。

| 需求分析用例 | 系统设计模块 | 系统实现界面 |
| --- | --- | --- |
| 用户登录与角色识别 | 用户登录与角色识别模块 | 登录注册界面、角色化主工作台 |
| 单张灾后图像检索 | 灾后图像检索模块 | 灾后图像检索界面、Top 5 结果展示 |
| 检索结果分析、反馈与报告导出 | 结果分析、反馈与报告模块 | 结果详情与前后对比界面、反馈标注、报告导出 |
| 检索历史与失败案例管理 | 检索历史与失败案例模块 | 检索历史界面、失败案例库界面 |
| 系统状态查看与索引管理 | 系统状态、索引与用户管理模块 | 系统状态与索引管理界面、用户管理界面 |

### 12.1 用户登录与角色识别界面实现

登录注册界面是系统入口界面。当用户未登录时，系统显示认证面板；用户登录成功后才进入主工作台。

界面功能：

- 登录已有账号。
- 注册新账号。
- 注册时通过下拉框选择检索用户、系统管理员或算法开发人员。
- 展示默认演示账号。
- 展示后端和索引状态。
- 登录失败时显示错误提示。

涉及状态变量：

| 变量 | 说明 |
| --- | --- |
| `authMode` | 当前认证模式，取值为 `login` 或 `register` |
| `authForm` | 用户名、密码、角色 |
| `token` | 登录后保存的会话 token |
| `user` | 当前登录用户 |
| `error` | 错误提示信息 |
| `health` | 后端健康状态 |

登录流程：

1. 用户输入用户名和密码。
2. 前端调用 `POST /api/auth/login`。
3. 后端校验用户名和密码。
4. 校验成功后，后端生成 token 并返回用户信息。
5. 前端保存 token 到 `localStorage`。
6. 前端调用 `bootstrapAfterAuth()` 加载系统状态、筛选项、样例和历史记录。
7. 界面切换为角色化工作台。

注册流程：

1. 用户切换到注册模式。
2. 输入用户名、密码并选择角色。
3. 前端调用 `POST /api/auth/register`。
4. 后端创建对应角色的本地用户并返回 token。
5. 前端自动进入系统。

对应后端接口：

| 接口 | 方法 | 功能 |
| --- | --- | --- |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/me` | GET | 根据 token 恢复会话 |
| `/api/auth/logout` | POST | 退出登录 |
| `/api/health` | GET | 查询系统状态 |

权限控制实现：

- 后端通过 `current_user()` 解析 Bearer token。
- 未登录用户访问检索、历史、反馈、管理接口时返回 401。
- 角色权限不足时返回 403。

默认演示账号：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `admin` | `admin123` | 系统管理员 |
| `developer` | `developer123` | 算法开发人员 |
| `retriever` | `retriever123` | 检索用户 |

#### 12.1.1 角色化主工作台实现

用户登录后进入主工作台。系统根据当前用户角色动态生成顶部导航标签。

界面功能：

- 显示当前用户名称和角色。
- 显示模型和索引是否在线。
- 支持刷新系统状态。
- 支持退出登录。
- 根据角色展示不同功能页面。

导航生成逻辑：

```text
retriever -> 检索、历史、报告导出
admin -> 检索、历史、失败案例、系统状态、索引管理、用户管理
developer -> 检索、历史、失败案例、系统状态、索引管理、算法分析、用户查看
```

涉及状态变量：

| 变量 | 说明 |
| --- | --- |
| `activeTab` | 当前选中的功能页 |
| `tabs` | 根据角色计算出的导航列表 |
| `canViewFailures` | 是否可查看失败案例 |
| `canViewUsers` | 是否可查看本地用户 |
| `canManageRetrievers` | 是否可维护检索用户 |
| `canOperateIndex` | 是否可执行索引维护 |
| `canDevelop` | 是否为算法开发人员 |

工作台刷新流程：

1. 调用 `/api/health` 获取检索服务状态。
2. 调用 `/api/options` 获取灾害类型和地区。
3. 调用 `/api/samples` 获取 hold 样例。
4. 调用 `/api/history` 获取历史记录。
5. 如果用户具备权限，额外加载失败案例、系统状态和用户列表。

### 12.2 灾后图像检索界面实现

检索界面是系统核心界面，对应“单张灾后图像检索”和“批量检索”功能。

界面组成：

- 灾害类型下拉框。
- 灾害地区下拉框。
- 候选过滤模式选择框。
- 排序聚合策略选择框。
- 图像上传区域。
- hold 样例列表。
- 查询图像预览。
- Top 5 灾前结果列表。

筛选参数：

| 参数 | 界面控件 | 说明 |
| --- | --- | --- |
| `disaster_type` | 灾害类型下拉框 | fire、flooding、wind |
| `disaster` | 灾害地区下拉框 | 具体灾害事件 |
| `filter_mode` | 候选过滤选择框 | both、type、disaster、none |
| `aggregation` | 排序策略选择框 | top_m、max、mean、sum、vote |

图像输入方式：

1. 用户上传单张灾后图像。
2. 用户多选图像进行批量检索。
3. 用户从 hold 样例列表中选择灾后 patch。

单图上传检索流程：

1. 用户选择一张图像。
2. 前端将图像和检索参数写入 `FormData`。
3. 调用 `POST /api/search`。
4. 后端保存上传图像到 `outputs/uploads/`。
5. 后端执行模型编码和 FAISS 检索。
6. 后端写入检索历史。
7. 前端展示 Top 5 结果。

样例路径检索流程：

1. 用户点击 hold 样例。
2. 前端保存样例的 `post_patch_path`。
3. 调用 `POST /api/search-by-path`。
4. 后端直接读取本地样例图像并检索。

批量检索流程：

1. 用户在上传区域多选图像。
2. 前端调用 `POST /api/batch-search`。
3. 后端逐张图像执行检索。
4. 每张图像单独保存历史记录。
5. 前端显示每个文件的成功或失败状态。
6. 用户可点击批量结果切换查看对应 Top 5。

对应后端接口：

| 接口 | 方法 | 功能 |
| --- | --- | --- |
| `/api/options` | GET | 获取筛选项 |
| `/api/samples` | GET | 获取样例图像 |
| `/api/search` | POST | 上传单图检索 |
| `/api/search-by-path` | POST | 本地路径检索 |
| `/api/batch-search` | POST | 批量上传检索 |
| `/api/batch-search-by-path` | POST | 批量路径检索 |
| `/api/assets` | GET | 读取本地图像资源 |

#### 12.2.1 Top 5 结果展示界面实现

检索完成后，系统在结果区展示 Top 5 灾前图像。

每个结果卡片展示：

- 排名。
- 灾前 patch 缩略图。
- 聚合相似度分数。
- best patch score。
- tile ID。
- damage label。
- patch hit count。
- tile patch count。

前端渲染数据来自后端返回的 `results` 数组。每个结果包含：

| 字段 | 说明 |
| --- | --- |
| `rank` | 排名 |
| `score` | 聚合分数 |
| `best_patch_score` | 最佳 patch 分数 |
| `tile_id` | tile 编号 |
| `positive_id` | 建筑正样本 ID |
| `building_uid` | 建筑 UID |
| `disaster` | 灾害地区 |
| `disaster_type` | 灾害类型 |
| `damage_label` | 损毁标签 |
| `pre_patch_url` | 灾前 patch 资源地址 |
| `pre_image_url` | 灾前 tile 资源地址 |
| `search_result_id` | 入库后的结果 ID |

用户点击结果卡片后，前端调用或读取结果详情，并在右侧详情面板中展示前后对比。

### 12.3 结果分析、反馈与报告界面实现

结果详情界面用于辅助检索用户判断检索结果是否正确，并为算法开发人员积累后续分析样本。

界面功能：

- 左侧展示查询灾后图像。
- 右侧展示命中灾前图像。
- 展示 tile、building、score、damage 等元信息。
- 支持填写反馈备注。
- 支持标记“正确”“错误”“不确定”。

详情数据来源：

- 如果结果来自当前检索响应，前端直接使用已有结果对象。
- 如果需要完整历史详情，前端调用 `GET /api/results/{result_id}`。

反馈提交流程：

1. 用户点击某个结果。
2. 用户填写备注。
3. 用户点击“正确”“错误”或“不确定”。
4. 前端调用 `POST /api/feedback`。
5. 后端写入 `feedback_records` 表。
6. 如果标记为错误，该记录会出现在失败案例库中。

### 12.4 检索历史与失败案例界面实现

历史界面用于展示已经完成的检索记录。

界面功能：

- 展示检索编号。
- 展示查询图像名称或路径。
- 展示灾害类型和地区。
- 展示 Top 1 分数。
- 展示检索时间。
- 点击历史记录可恢复查看该次检索的结果。
- 支持导出 Markdown 报告压缩包。
- 支持单选、多选或全选历史记录并导出 Markdown 报告压缩包，压缩包内同时包含对应图像。

权限规则：

- `retriever` 只能查看本人历史。
- `admin`、`developer` 可以查看全部历史。

历史加载流程：

1. 前端进入历史页面。
2. 调用 `GET /api/history`。
3. 后端根据用户角色过滤数据。
4. 前端渲染历史列表。
5. 用户点击记录后调用 `GET /api/history/{search_id}`。
6. 前端将历史结果恢复到结果展示区域。

报告导出流程：

1. 用户选择某条历史记录。
2. 点击“导出 Markdown 报告”。
3. 前端调用 `GET /api/history/{search_id}/report`。
4. 后端根据历史记录生成 Markdown 报告和对应图像，并打包为 zip。
5. 前端将响应保存为压缩包文件。

Markdown 报告包导出流程：

1. 用户在历史列表中勾选一条或多条记录，也可以点击全选。
2. 前端调用 `GET /api/history/export?ids=...`。
3. 后端根据当前角色过滤可访问记录，`retriever` 只能导出本人历史。
4. 后端为每条记录生成 `report.md`，并拷贝查询图像和对应灾前图像。
5. 前端保存为 `retrieval_history.zip`。

#### 12.4.1 失败案例库界面实现

失败案例库面向系统管理员和算法开发人员，用于查看和分析检索效果较差的样本。

进入条件：

- 用户角色为 `admin` 或 `developer`。

界面功能：

- 展示低分检索记录。
- 展示被用户标记为错误的检索记录。
- 点击失败案例可打开复核弹窗，显示灾后查询图像和灾前候选图像，便于再次判断。
- 支持管理员在失败案例列表中修改结果反馈。
- 支持单选、多选或全选失败案例并导出 Markdown 报告压缩包。
- 用于后续模型改进和论文案例分析。

后端判断规则：

- `top_score` 低于指定阈值的检索记录。
- 或存在 `wrong` 反馈的检索结果。

对应接口：

```text
GET /api/failures
GET /api/failures/export
```

#### 12.4.2 算法分析界面实现

算法分析界面仅面向算法开发人员，用于把失败案例、反馈标注和检索策略统计转化为模型优化依据。

界面功能：

- 汇总检索次数、平均 Top 分数、低分记录数和错误反馈数。
- 按灾害类型统计检索效果。
- 按过滤模式和聚合策略统计平均分数与耗时。
- 展示最近失败样本，支持打开失败案例复核弹窗。
- 读取本地 `checkpoints/` 下的模型文件并支持切换当前 checkpoint。
- 支持导出选中的失败案例 Markdown 报告压缩包。

对应接口：

```text
GET /api/algorithm/analysis
GET /api/models
POST /api/models/switch
```

权限规则：

- `developer` 可以访问。
- `retriever` 和 `admin` 访问时返回 403。

### 12.5 系统状态、索引与用户管理界面实现

系统状态界面面向系统管理员和算法开发人员，用于查看本地部署状态。

界面功能：

- 展示推理设备。
- 展示图库 patch 数和 tile 数。
- 展示索引任务状态。
- 展示 checkpoint、features、metadata、source_csv、sqlite 等文件是否存在。
- 管理员和算法开发人员可重载索引。
- 管理员和算法开发人员可触发完整 hold 索引重建。

状态数据来源：

```text
GET /api/system/status
GET /api/index/status
```

索引重载流程：

1. 管理员点击“重载索引”。
2. 前端调用 `POST /api/index/reload`。
3. 后端重新执行模型和索引加载。
4. 前端刷新系统状态。

索引重建流程：

1. 管理员点击“重建完整 hold 索引”。
2. 前端调用 `POST /api/index/rebuild`。
3. 后端使用后台进程执行 `scripts/build_api_index.py`。
4. 前端展示任务状态、进程号和日志路径。

#### 12.5.1 用户管理界面实现

用户管理界面面向系统管理员和算法开发人员，但两类角色的能力不同：管理员不显示 developer 用户信息，只能对检索用户进行增删改查；算法开发人员可以查看全部用户信息，但不负责修改账号。

界面功能：

- 管理员展示非 developer 用户，开发人员展示全部用户。
- 展示用户编号、用户名、角色和最近登录时间。
- 支持管理员新增检索用户。
- 支持管理员按用户名查找检索用户。
- 支持管理员修改检索用户的用户名和密码。
- 支持管理员删除或停用检索用户。
- 支撑答辩时展示系统的多角色管理能力。

对应接口：

```text
GET /api/users
POST /api/users
PATCH /api/users/{user_id}
DELETE /api/users/{user_id}
```

权限规则：

- `admin` 可以查看非 developer 用户，并维护 `retriever` 用户。
- `developer` 可以查看全部用户，但不提供修改入口。
- `retriever` 访问时返回 403。

#### 12.5.2 图像资源显示实现

前端显示图像时并不直接读取本地文件路径，而是通过后端资源接口：

```text
GET /api/assets?path=...
```

这样做有两个目的：

1. 统一前端图片加载方式，避免浏览器直接访问本地文件失败。
2. 后端可以限制资源访问范围，只允许读取仓库目录内文件，防止任意文件读取。

系统中样例图像、查询图像、灾前结果图像、配对灾后图像均通过该接口显示。

## 13. 本章小结

本章从代码结构、后端服务、索引构建、前端实现和界面功能等方面说明了灾后检索系统的具体实现。系统不仅实现了基本的灾后图像检索功能，还补充了登录注册、角色化工作台、检索历史、结果详情、前后对比、报告导出、用户反馈、失败案例库、批量检索、系统状态和用户管理等应用功能。

从实现角度看，系统前端负责组织用户交互和状态展示，后端负责账号鉴权、检索服务、历史记录和资源访问，检索模型负责灾后到灾前的向量匹配，SQLite 负责本地业务数据持久化。各界面功能均与后端接口对应，形成了一个完整的本地离线灾后检索应用系统。
