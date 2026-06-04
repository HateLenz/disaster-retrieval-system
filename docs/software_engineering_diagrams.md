# 软件工程图

本文档给出灾后图像检索系统的主要软件工程图。系统服务于灾情研判场景，但权限模型只设置三类系统角色：检索用户 `retriever`、系统管理员 `admin`、算法开发人员 `developer`。

## 1. 总体用例图

```mermaid
flowchart LR
    Retriever[检索用户 retriever]
    Admin[系统管理员 admin]
    Developer[算法开发人员 developer]

    UC_AUTH((登录注册与角色识别))
    UC_SEARCH((灾后图像检索))
    UC_RESULT((结果查看/反馈/报告))
    UC_HISTORY((检索历史))
    UC_FAILURE((失败案例库))
    UC_SYSTEM((系统状态查看))
    UC_INDEX((索引重载与重建))
    UC_USERS((用户管理/用户查看))
    UC_ALGO((算法分析))
    UC_MODEL((模型切换))

    Retriever --> UC_AUTH
    Retriever --> UC_SEARCH
    Retriever --> UC_RESULT
    Retriever --> UC_HISTORY

    Admin --> UC_AUTH
    Admin --> UC_SEARCH
    Admin --> UC_RESULT
    Admin --> UC_HISTORY
    Admin --> UC_FAILURE
    Admin --> UC_SYSTEM
    Admin --> UC_INDEX
    Admin --> UC_USERS

    Developer --> UC_AUTH
    Developer --> UC_SEARCH
    Developer --> UC_RESULT
    Developer --> UC_HISTORY
    Developer --> UC_FAILURE
    Developer --> UC_SYSTEM
    Developer --> UC_INDEX
    Developer --> UC_ALGO
    Developer --> UC_USERS
    Developer --> UC_MODEL
```

说明：

- 检索用户负责业务使用，完成灾后图像检索、结果查看、反馈和报告导出。
- 系统管理员负责系统运行维护，包括检索用户增删改查、系统状态、索引重载和索引重建；管理员不查看 developer 用户信息。
- 算法开发人员负责模型效果分析，包括失败案例、算法分析、模型切换和检索策略调试，并可查看全部用户信息。

## 2. 模块结构图

```mermaid
flowchart TB
    subgraph Frontend[Electron + Vue 前端]
        AuthView[登录注册界面]
        SearchView[检索工作台]
        HistoryView[历史与报告]
        FailureView[失败案例库]
        SystemView[系统状态与索引]
        UserView[用户管理]
        AlgoView[算法分析]
    end

    subgraph API[FastAPI 后端]
        AuthAPI[认证与权限 API]
        SearchAPI[检索 API]
        HistoryAPI[历史/反馈/报告 API]
        AdminAPI[系统/索引/用户 API]
        AlgoAPI[算法分析 API]
    end

    subgraph Service[本地服务层]
        Store[LocalAppStore]
        Retrieval[DisasterRetrievalService]
        IndexBuilder[build_api_index.py]
    end

    subgraph Data[本地数据]
        AppDB[(retrieval_app.db)]
        GalleryDB[(retrieval_hold_stage1_v2.db)]
        Features[(pre_features.npy/faiss)]
        Metadata[(pre_metadata.csv)]
        Images[xBD building patches]
        Checkpoint[clip_visual_baseline_smoke_stage1_v2.pt]
    end

    AuthView --> AuthAPI
    SearchView --> SearchAPI
    HistoryView --> HistoryAPI
    FailureView --> HistoryAPI
    SystemView --> AdminAPI
    UserView --> AdminAPI
    AlgoView --> AlgoAPI

    AuthAPI --> Store
    HistoryAPI --> Store
    AdminAPI --> Store
    AlgoAPI --> Store
    SearchAPI --> Retrieval
    AdminAPI --> Retrieval
    AdminAPI --> IndexBuilder

    Store --> AppDB
    Retrieval --> GalleryDB
    Retrieval --> Features
    Retrieval --> Metadata
    Retrieval --> Images
    Retrieval --> Checkpoint
```

## 3. 权限控制图

```mermaid
flowchart LR
    Request[前端请求]
    Token[Bearer token]
    User[当前用户]
    Role{角色判断}

    Request --> Token
    Token --> User
    User --> Role

    Role -->|retriever| R[检索/本人历史/反馈/报告]
    Role -->|admin| A[全部历史/失败案例/系统状态/索引管理/检索用户管理]
    Role -->|developer| D[全部历史/失败案例/系统状态/索引管理/算法分析/模型切换/全部用户查看]
```

核心规则：

- `/api/users` 允许 `admin` 和 `developer`，其中 `admin` 不返回 developer 用户，`developer` 返回全部用户。
- `/api/users` 的新增、修改和删除操作仅允许 `admin`，且只能操作 `retriever` 用户。
- `/api/failures`、`/api/system/status`、`/api/index/*` 允许 `admin` 和 `developer`。
- `/api/algorithm/analysis`、`/api/models` 和 `/api/models/switch` 仅允许 `developer`。
- `retriever` 只能查看本人历史，`admin` 和 `developer` 可以查看全部历史。

## 4. 登录注册时序图

```mermaid
sequenceDiagram
    actor U as 用户
    participant V as Vue 前端
    participant A as FastAPI
    participant S as LocalAppStore
    participant DB as SQLite

    U->>V: 输入用户名和密码
    V->>A: POST /api/auth/login
    A->>S: authenticate(username,password)
    S->>DB: 查询用户与密码摘要
    DB-->>S: 用户记录
    S-->>A: 公共用户信息
    A->>S: create_session(user_id)
    S->>DB: 写入 sessions
    A-->>V: token + user + roles
    V->>V: 保存 token 并生成角色化导航
```

## 5. 灾后图像检索时序图

```mermaid
sequenceDiagram
    actor U as 检索用户
    participant V as Vue 前端
    participant A as FastAPI
    participant R as DisasterRetrievalService
    participant F as FAISS/NumPy
    participant S as LocalAppStore

    U->>V: 选择灾害类型、地区并上传灾后图像
    V->>A: POST /api/search
    A->>R: search_bytes(image, filters)
    R->>R: 图像预处理和 CLIP 编码
    R->>F: 向量相似性检索
    F-->>R: 候选行号和相似度
    R-->>A: Top 5 灾前图像和元信息
    A->>S: create_search_record()
    S-->>A: search_id/result_id
    A-->>V: 检索结果
    V-->>U: 展示 Top 5 和前后对比
```

## 6. 失败案例与算法分析时序图

```mermaid
sequenceDiagram
    actor D as 算法开发人员
    participant V as Vue 前端
    participant A as FastAPI
    participant S as LocalAppStore
    participant DB as SQLite

    D->>V: 打开算法分析
    V->>A: GET /api/algorithm/analysis
    A->>A: 校验 developer 角色
    A->>S: algorithm_analysis()
    S->>DB: 统计检索记录、反馈和失败样本
    DB-->>S: 统计结果
    S-->>A: 平均分、低分数、反馈数、策略统计
    A-->>V: 分析结果
    V-->>D: 展示模型效果与策略表现
```

## 7. 索引维护时序图

```mermaid
sequenceDiagram
    actor A0 as 系统管理员/算法开发人员
    participant V as Vue 前端
    participant A as FastAPI
    participant R as DisasterRetrievalService
    participant P as build_api_index.py
    participant FS as 本地文件

    A0->>V: 点击重建完整 hold 索引
    V->>A: POST /api/index/rebuild
    A->>A: 校验 admin/developer 角色
    A->>P: 后台执行索引构建脚本
    P->>FS: 写入 pre_features/faiss/metadata/db
    A-->>V: 返回 pid 和日志路径
    A0->>V: 点击重载索引
    V->>A: POST /api/index/reload
    A->>R: load()
    R->>FS: 重新加载模型和索引
    A-->>V: 返回最新状态
```

## 8. 模块与接口对应表

| 模块 | 前端入口 | 后端接口 | 主要数据 |
| --- | --- | --- | --- |
| 用户登录与角色识别 | 登录/注册界面 | `/api/auth/*` | `users`、`sessions` |
| 灾后图像检索 | 检索工作台 | `/api/search`、`/api/search-by-path` | 模型、FAISS、图库元信息 |
| 结果分析、反馈与报告 | 结果详情、历史报告 | `/api/results/*`、`/api/feedback`、`/api/history/*/report` | `search_results`、`feedback_records` |
| 检索历史与失败案例 | 历史、失败案例库 | `/api/history`、`/api/failures` | `search_records`、`feedback_records` |
| 系统状态、索引与用户管理 | 系统、用户管理 | `/api/system/status`、`/api/index/*`、`/api/users` | 索引文件、用户表 |
| 算法分析 | 算法分析、模型切换 | `/api/algorithm/analysis`、`/api/models`、`/api/models/switch` | 历史记录、反馈记录、失败案例、checkpoint |
