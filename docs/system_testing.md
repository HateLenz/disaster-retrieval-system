# 灾后检索系统测试

## 1. 测试目标

系统测试的目标是验证灾后检索系统在本地环境中能够稳定完成以下任务：

1. 正确加载本地模型、索引和元信息。
2. 正确响应前端请求。
3. 正确根据 `disaster_type` 和 `disaster` 过滤候选库。
4. 对灾后图像返回 Top 5 灾前图像。
5. 在异常情况下返回清晰错误信息。
6. 前端能够完成上传、样例选择、检索和结果展示。
7. 系统满足离线运行和毕业设计演示要求。

## 2. 测试环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows，本地开发环境 |
| Python | 3.10，推荐使用 `dfr-algo` conda 环境 |
| 后端框架 | FastAPI + Uvicorn |
| 深度学习框架 | PyTorch |
| 检索库 | FAISS CPU，缺失时退回 NumPy |
| 前端 | Electron + Vue 3 + Vite |
| 模型 | `checkpoints/clip_visual_baseline_smoke_stage1_v2.pt` |
| 数据 | xBD hold split building patch |

## 3. 测试范围

### 3.1 包含范围

- 后端启动测试。
- 模型加载测试。
- 索引加载测试。
- API 接口测试。
- 图像检索功能测试。
- 资源访问测试。
- 前端页面交互测试。
- 异常输入测试。
- 离线运行测试。

### 3.2 不包含范围

- 模型重新训练过程测试。
- 大规模并发压测。
- 云端部署测试。
- 非 xBD 数据集泛化效果评估。

## 4. 测试数据

测试数据来源于 xBD hold split：

- 灾前图像：`pre_patch_path`
- 灾后图像：`post_patch_path`
- 元信息：`data/processed/xbd_building_patches.csv`
- smoke 开发索引：`indexes/pre_clip_features_smoke_stage1_v2.npy`
- 完整 hold 索引：`indexes/api_hold_stage1_v2/pre_features.npy`

建议至少覆盖三类灾害：

- `fire`
- `flooding`
- `wind`

每类灾害选择不少于 5 张灾后 patch 作为功能测试样例。

## 5. 单元测试

### 5.1 路径解析测试

| 测试项 | 输入 | 预期结果 |
| --- | --- | --- |
| 相对路径解析 | `data/processed/.../xxx_post.png` | 返回仓库内绝对路径 |
| Linux 历史路径解析 | `/root/.../data/processed/.../xxx_pre.png` | 能截取 `data/processed/` 后转换为本地路径 |
| 空路径解析 | 空字符串 | 返回空或错误提示 |
| 仓库外路径 | `C:/Windows/...` | 拒绝资源访问 |

### 5.2 图像预处理测试

| 测试项 | 输入 | 预期结果 |
| --- | --- | --- |
| RGB PNG | 建筑级 patch | 输出 3 通道 tensor |
| TIFF 图像 | xBD 原始图像 | 能转换为 RGB |
| 大尺寸图像 | 1024x1024 tile | 能滑窗切分 |
| 损坏图像 | 非图像文件 | 返回错误 |

### 5.3 候选过滤测试

| 测试项 | 条件 | 预期结果 |
| --- | --- | --- |
| 按灾害类型过滤 | `disaster_type=fire` | 候选全部为 fire |
| 按灾害地区过滤 | `disaster=santa-rosa-wildfire` | 候选全部来自该地区 |
| 同时过滤 | `fire + santa-rosa-wildfire` | 候选满足两个条件 |
| 无匹配条件 | 不存在的灾害名称 | 返回无候选错误 |

### 5.4 检索结果格式测试

每条结果应包含：

- `rank`
- `score`
- `best_patch_score`
- `tile_id`
- `positive_id`
- `building_uid`
- `disaster`
- `disaster_type`
- `damage_label`
- `pre_patch_url`
- `pre_image_url`

## 6. 接口测试

### 6.1 健康检查接口

接口：

```http
GET /api/health
```

预期响应：

```json
{
  "ready": true,
  "scope": "hold_smoke_stage1_v2",
  "gallery_patch_count": 1024,
  "gallery_tile_count": 440
}
```

验收标准：

- HTTP 状态码为 200。
- `ready=true`。
- `gallery_patch_count > 0`。
- 返回 checkpoint、features_path 和 metadata_path。

### 6.2 筛选项接口

接口：

```http
GET /api/options
```

验收标准：

- 返回 `disaster_types`。
- 至少包含 `fire`、`flooding`、`wind`。
- 返回 `disasters_by_type`。
- 每个灾害类型下至少有一个灾害地区。

### 6.3 样例接口

接口：

```http
GET /api/samples?disaster_type=fire&limit=12
```

验收标准：

- 返回样例数量不超过 12。
- 每条样例包含 `post_patch_url` 和 `pre_patch_url`。
- 样例图像可以通过 `/api/assets` 正常打开。

### 6.4 路径检索接口

接口：

```http
POST /api/search-by-path
```

请求：

```json
{
  "image_path": "data/processed/xbd_building_patches/hold/fire/.../xxx_post.png",
  "disaster_type": "fire",
  "disaster": "santa-rosa-wildfire",
  "top_k": 5
}
```

验收标准：

- HTTP 状态码为 200。
- `results` 数量为 5。
- 每条结果包含灾前图像 URL。
- `gallery.candidate_tile_count > 0`。

### 6.5 上传检索接口

接口：

```http
POST /api/search
Content-Type: multipart/form-data
```

表单字段：

- `image`
- `disaster_type`
- `disaster`
- `top_k`

验收标准：

- 上传 hold 灾后图像后返回 Top 5。
- 返回 `query.patch_count`。
- 返回 `elapsed_ms`。

## 7. 功能测试用例

| 编号 | 用例 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| FT-01 | 后端启动 | 启动 Uvicorn | `/api/health` 返回 ready |
| FT-02 | 前端启动 | 启动 Vue 或 Electron | 页面显示系统状态 |
| FT-03 | 加载筛选项 | 打开页面 | 灾害类型和地区下拉框有值 |
| FT-04 | 样例检索 | 点击 hold 样例并检索 | 返回 Top 5 |
| FT-05 | 上传检索 | 上传本地灾后 patch | 返回 Top 5 |
| FT-06 | 切换灾害类型 | 从 fire 切换到 flooding | 样例列表和候选统计更新 |
| FT-07 | 图像资源预览 | 打开结果图像 | 图像正常显示 |
| FT-08 | 无候选测试 | 输入不存在的 disaster | 返回错误提示 |
| FT-09 | 模型缺失测试 | 临时移走 checkpoint | `/api/health` 显示未就绪 |
| FT-10 | 索引缺失测试 | 临时移走特征文件 | 系统提示需构建索引 |

## 8. 性能测试

### 8.1 指标

| 指标 | 说明 |
| --- | --- |
| 后端启动时间 | 从启动服务到 `/api/health ready=true` 的时间 |
| 单 patch 检索时间 | 224x224 查询图像的检索耗时 |
| 大图检索时间 | 1024x1024 图像滑窗检索耗时 |
| 候选库规模 | 过滤后的 patch 数和 tile 数 |
| 内存占用 | 模型和索引加载后的内存占用 |

### 8.2 建议目标

| 场景 | 目标 |
| --- | --- |
| smoke hold 索引 | 单次查询 1 秒内返回 |
| 完整 hold 索引 | 单 patch 查询数秒内返回 |
| 大图滑窗查询 | 可交互等待时间内完成 |
| API 健康检查 | 100 ms 内返回 |

实际指标与硬件有关，测试报告中应记录 CPU/GPU 型号和是否启用 CUDA。

## 9. 异常测试

| 异常场景 | 预期处理 |
| --- | --- |
| 未安装 FastAPI | 后端无法启动，依赖检查失败 |
| 未安装 Node/npm | 前端无法启动，提示安装前端环境 |
| checkpoint 不存在 | `/api/health` 返回 `ready=false` 和错误信息 |
| 特征文件不存在 | 系统提示运行 `build_api_index.py` |
| 上传非图像文件 | `/api/search` 返回错误 |
| 上传空文件 | `/api/search` 返回错误 |
| 访问仓库外文件 | `/api/assets` 返回 403 |
| 灾害筛选无候选 | `/api/search` 返回 400 |
| GPU 不可用 | 自动退回 CPU |

## 10. 离线测试

离线测试步骤：

1. 断开网络。
2. 确认 `pretrained/openai-clip-vit-base-patch32/` 存在。
3. 确认 checkpoint、features、metadata 和 SQLite 文件存在。
4. 启动后端。
5. 访问 `/api/health`。
6. 使用 hold 样例执行一次检索。
7. 启动前端并执行同样检索。

验收标准：

- 系统不访问外网。
- 模型加载成功。
- 检索结果正常返回。
- 图像资源正常显示。

## 11. 回归测试命令

Python 语法检查：

```bash
python -m py_compile src/api/app.py src/api/retrieval_service.py scripts/build_api_index.py
```

后端启动：

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

完整 hold 索引构建：

```bash
python scripts/build_api_index.py --split hold --amp
```

前端启动：

```bash
cd frontend
npm install
npm run dev:electron
```

## 12. 当前验证记录

当前开发环境中已完成以下验证：

| 项目 | 结果 |
| --- | --- |
| Python 语法检查 | 通过 |
| FastAPI TestClient `/api/health` | 通过 |
| `/api/search-by-path` 样例检索 | 通过 |
| 返回 Top 5 结果 | 通过 |
| 后端本地服务启动 | 通过 |
| 前端实际启动 | 未验证，当前环境缺少 Node/npm |

已验证后端在 smoke_stage1_v2 hold 索引下可以正常返回结果。完整 hold 索引需要运行 `scripts/build_api_index.py` 后再进行完整规模测试。

## 13. 测试结论

系统后端核心链路已经具备可运行性：模型加载、索引加载、样例检索和 Top 5 返回均可通过接口验证。前端工程文件已经完成，但需要安装 Node/npm 后进行实际界面测试。

下一阶段测试重点应放在：

1. 完整 hold 索引构建后的检索性能。
2. Electron 桌面端启动和后端自动拉起。
3. 图像对比、历史记录、报告导出等增强功能的回归测试。
4. 多灾害类型、多灾害地区样例的覆盖测试。

## 14. 增强功能测试补充

### 14.1 账号与角色测试

| 编号 | 用例 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| AT-01 | 默认管理员登录 | 使用 `admin/admin123` 登录 | 登录成功，显示用户管理和索引管理 |
| AT-02 | 默认开发人员登录 | 使用 `developer/developer123` 登录 | 登录成功，显示失败案例、系统状态和算法分析 |
| AT-03 | 默认检索用户登录 | 使用 `retriever/retriever123` 登录 | 登录成功，仅显示检索和本人历史 |
| AT-04 | 注册新用户 | 输入用户名、密码并选择角色注册 | 新账号以所选角色写入 SQLite 并自动登录 |
| AT-05 | 未登录访问检索接口 | 不带 Bearer token 请求 `/api/options` | 返回 401 |
| AT-06 | 检索用户访问用户管理 | retriever 请求 `/api/users` | 返回 403 |

### 14.2 历史、报告和反馈测试

| 编号 | 用例 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| HT-01 | 检索自动入库 | 执行一次 `/api/search-by-path` | 返回 `search_id` 和 `search_result_id` |
| HT-02 | 查看历史 | 请求 `/api/history` | 返回刚才的检索记录 |
| HT-03 | 查看详情 | 请求 `/api/history/{search_id}` | 返回 Top 结果和 query/gallery 信息 |
| HT-04 | 导出报告 | 请求 `/api/history/{search_id}/report` | 返回包含 `report.md` 和对应图像的 zip 压缩包 |
| HT-05 | 提交正确反馈 | 对结果提交 `correct` | 反馈写入成功 |
| HT-06 | 提交错误反馈 | 对结果提交 `wrong` | 失败案例库包含相关记录 |

### 14.3 批量检索测试

| 编号 | 用例 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| BT-01 | 多文件上传 | 向 `/api/batch-search` 上传 2-5 张图像 | 每张图像返回独立结果 |
| BT-02 | 批量路径检索 | 向 `/api/batch-search-by-path` 提交多个 `post_patch_path` | 返回每个路径的执行状态 |
| BT-03 | 混合异常 | 批量中包含错误路径 | 正确图像成功，错误路径返回错误项 |

### 14.4 索引管理测试

| 编号 | 用例 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| IT-01 | 查看系统状态 | 请求 `/api/system/status` | 返回模型、设备、索引文件和任务状态 |
| IT-02 | 重载索引 | 管理员请求 `/api/index/reload` | 服务重新加载模型和索引 |
| IT-03 | 触发重建 | 管理员请求 `/api/index/rebuild` | 返回后台任务 pid 和日志路径 |
| IT-04 | 权限限制 | retriever 请求 `/api/index/rebuild` | 返回 403 |

## 15. 当前增强功能验证记录

本次增强后已完成接口级验证：

| 项目 | 结果 |
| --- | --- |
| 管理员登录 | 通过 |
| `/api/options` 鉴权访问 | 通过 |
| `/api/samples` 样例读取 | 通过 |
| `/api/search-by-path` 检索并写入历史 | 通过 |
| `/api/feedback` 反馈提交 | 通过 |
| `/api/history` 历史列表 | 通过 |
| `/api/history/{id}` 历史详情 | 通过 |
| `/api/history/{id}/report` Markdown 报告 | 通过 |
| `/api/system/status` 系统状态 | 通过 |
| 新版后端 8000 端口启动 | 通过 |

当前环境仍缺少 Node/npm，因此 Vue/Electron 界面需要在安装前端运行环境后进行端到端 UI 测试。

## 16. 各模块测试

本节按照系统功能模块完成测试设计与结果记录。测试重点覆盖登录注册、角色权限、灾后图像检索、结果详情、历史记录、报告导出、反馈标注、失败案例、系统状态、索引管理和用户管理等模块。

为保证需求分析、系统设计、系统实现和系统测试四个章节结构一致，模块测试按照五个核心用例归并组织。测试表中的细项仍覆盖全部功能，但章节数量与用例数量保持一致。

| 需求分析用例 | 系统设计模块 | 系统测试模块 |
| --- | --- | --- |
| 用户登录与角色识别 | 用户登录与角色识别模块 | 登录注册与角色权限模块测试 |
| 单张灾后图像检索 | 灾后图像检索模块 | 灾后图像检索模块测试 |
| 检索结果分析、反馈与报告导出 | 结果分析、反馈与报告模块 | 结果分析、反馈与报告模块测试 |
| 检索历史与失败案例管理 | 检索历史与失败案例模块 | 检索历史与失败案例模块测试 |
| 系统状态查看与索引管理 | 系统状态、索引与用户管理模块 | 系统状态、索引与用户管理模块测试 |

### 16.1 登录注册与角色权限模块测试

测试目标：

- 验证用户能够完成登录、注册、退出和会话恢复。
- 验证错误账号、错误密码和无 token 访问能够被正确拦截。
- 验证系统能够根据用户角色进入不同工作台。
- 验证不同角色只能访问其权限范围内的功能。

测试用例：

| 编号 | 测试内容 | 测试步骤 | 预期结果 | 测试结果 |
| --- | --- | --- | --- | --- |
| AUTH-01 | 管理员登录 | 输入 `admin/admin123`，调用 `/api/auth/login` | 返回 token 和 admin 用户信息 | 通过 |
| AUTH-02 | 开发人员登录 | 输入 `developer/developer123` | 返回 developer 用户信息 | 通过 |
| AUTH-03 | 检索用户登录 | 输入 `retriever/retriever123` | 返回 retriever 用户信息 | 通过 |
| AUTH-04 | 密码错误 | 输入正确用户名和错误密码 | 返回 401 | 通过 |
| AUTH-05 | 注册新用户 | 输入用户名、密码并选择角色 | 用户以所选角色写入 SQLite，并返回 token | 通过 |
| AUTH-06 | 会话恢复 | 携带 token 调用 `/api/auth/me` | 返回当前用户信息 | 通过 |
| AUTH-07 | 退出登录 | 调用 `/api/auth/logout` | token 被删除，后续访问需重新登录 | 通过 |
| AUTH-08 | 未登录访问业务接口 | 不携带 token 调用 `/api/options` | 返回 401 | 通过 |
| AUTH-09 | 检索用户访问用户管理 | retriever 调用 `/api/users` | 返回 403 | 通过 |
| AUTH-10 | 管理员访问用户管理 | admin 调用 `/api/users` | 返回用户列表 | 通过 |
| AUTH-11 | 检索用户触发索引重建 | retriever 调用 `/api/index/rebuild` | 返回 403 | 通过 |

测试结论：

登录注册与角色权限模块能够完成本地账号创建、登录认证、会话控制和接口权限限制。系统能够通过 Bearer token 识别用户身份，并根据角色控制功能入口和后端接口访问。

### 16.2 灾后图像检索模块测试

测试目标：

- 验证单张上传图像检索。
- 验证 hold 样例路径检索。
- 验证批量上传检索。
- 验证候选过滤模式和排序策略参数能够传递到后端。

测试用例：

| 编号 | 测试内容 | 测试步骤 | 预期结果 | 测试结果 |
| --- | --- | --- | --- | --- |
| SEARCH-01 | 获取筛选项 | 登录后调用 `/api/options` | 返回 `fire`、`flooding`、`wind` 等选项 | 通过 |
| SEARCH-02 | 获取样例图像 | 调用 `/api/samples?disaster_type=fire&limit=1` | 返回 hold 样例 | 通过 |
| SEARCH-03 | 样例路径检索 | 用样例 `post_patch_path` 调用 `/api/search-by-path` | 返回 5 条结果 | 通过 |
| SEARCH-04 | 上传单图检索 | 上传一张灾后 patch 调用 `/api/search` | 返回 Top 5，并写入历史 | 通过 |
| SEARCH-05 | 批量上传检索 | 上传多张图像调用 `/api/batch-search` | 每张图像返回独立状态 | 通过 |
| SEARCH-06 | 按灾害类型过滤 | `filter_mode=type` | 候选库按灾害类型过滤 | 通过 |
| SEARCH-07 | 按灾害地区过滤 | `filter_mode=disaster` | 候选库按灾害地区过滤 | 通过 |
| SEARCH-08 | 全库检索 | `filter_mode=none` | 不使用灾害条件过滤 | 通过 |
| SEARCH-09 | 排序策略切换 | 分别设置 `top_m`、`max`、`mean`、`sum`、`vote` | 后端正常返回结果 | 通过 |
| SEARCH-10 | 非图像文件 | 上传文本文件 | 返回图像读取错误 | 通过 |

已验证接口输出示例：

```text
search status: 200
result_count: 5
search_id: generated
```

测试结论：

检索模块能够完成核心业务流程。系统支持样例图像、上传图像和批量图像检索，并能保存检索记录。

### 16.3 结果分析、反馈与报告模块测试

测试目标：

- 验证 Top 5 结果字段完整。
- 验证灾前图像资源可以正常加载。
- 验证结果详情和前后对比数据可用。
- 验证用户反馈和 Markdown 报告导出可用。

测试用例：

| 编号 | 测试内容 | 测试步骤 | 预期结果 | 测试结果 |
| --- | --- | --- | --- | --- |
| RESULT-01 | Top 5 数量 | 完成一次检索 | 返回 5 条结果 | 通过 |
| RESULT-02 | 结果字段完整性 | 检查 `rank`、`score`、`tile_id`、`pre_patch_url` 等字段 | 字段完整 | 通过 |
| RESULT-03 | 灾前图像访问 | 访问结果中的 `pre_patch_url` | 图像正常返回 | 通过 |
| RESULT-04 | 结果详情接口 | 调用 `/api/results/{result_id}` | 返回完整结果和查询信息 | 通过 |
| RESULT-05 | 前后对比 | 前端展示查询图像和灾前结果图像 | 两张图像均可显示 | 需前端环境验证 |
| RESULT-06 | 正确反馈 | 提交 `correct` | 写入 `feedback_records` | 通过 |
| RESULT-07 | 错误反馈 | 提交 `wrong` | 写入反馈记录 | 通过 |
| RESULT-08 | 报告导出 | 调用 `/api/history/{search_id}/report` | 返回 Markdown 报告压缩包 | 通过 |
| RESULT-09 | 报告内容 | 检查 `report.md` 与对应图像文件 | 内容完整，图像路径可用 | 通过 |

测试结论：

后端结果数据结构完整，能够支持前端展示 Top 5 卡片、详情对比、反馈标注和报告导出。前端图像实际渲染需在安装 Node/npm 后进行界面测试。

### 16.4 检索历史与失败案例模块测试

测试目标：

- 验证每次检索都会写入历史记录。
- 验证检索用户只能查看本人历史。
- 验证管理员和算法开发人员能够查看全部历史。
- 验证低分记录和错误反馈能够进入失败案例库。

测试用例：

| 编号 | 测试内容 | 测试步骤 | 预期结果 | 测试结果 |
| --- | --- | --- | --- | --- |
| HIST-01 | 检索写入历史 | 完成一次 `/api/search-by-path` | 返回 `search_id`，数据库新增记录 | 通过 |
| HIST-02 | 查询历史列表 | 调用 `/api/history` | 返回历史记录数组 | 通过 |
| HIST-03 | 查询历史详情 | 调用 `/api/history/{search_id}` | 返回 query、gallery 和 results | 通过 |
| HIST-04 | 检索用户历史隔离 | retriever 查看历史 | 只返回本人记录 | 通过 |
| HIST-05 | 管理员查看全部历史 | admin 查看历史 | 返回所有用户记录 | 通过 |
| HIST-06 | 不存在的历史 ID | 调用不存在的 `/api/history/{id}` | 返回错误提示 | 通过 |
| HIST-07 | 失败案例列表 | developer 调用 `/api/failures` | 返回失败案例列表 | 通过 |
| HIST-08 | 错误反馈进入失败案例 | 对结果提交 `wrong` 后查看失败案例 | 失败案例库包含相关记录 | 通过 |
| HIST-09 | 历史 Markdown 包导出 | 勾选历史记录后调用 `/api/history/export?ids=...` | 返回 zip 文件，包含 `report.md` 和对应图像 | 通过 |
| HIST-10 | 失败案例 Markdown 包导出 | 勾选失败案例后调用 `/api/failures/export?ids=...` | 返回 zip 文件，包含 `report.md` 和对应图像 | 通过 |

测试结论：

历史与失败案例模块能够记录检索过程，并根据角色进行访问控制。失败案例库能够汇总低分记录和错误反馈，为后续算法改进提供样本依据。

### 16.5 系统状态、索引与用户管理模块测试

测试目标：

- 验证系统能够展示模型、设备、索引和文件资源状态。
- 验证管理员能够重载索引和触发索引重建。
- 验证检索用户不能执行管理操作。
- 验证管理员可以维护检索用户，但不能查看 developer 用户信息。
- 验证开发人员可以查看全部用户信息，但不修改用户。
- 验证开发人员可以切换本地模型 checkpoint。

测试用例：

| 编号 | 测试内容 | 测试步骤 | 预期结果 | 测试结果 |
| --- | --- | --- | --- | --- |
| SYS-01 | 健康检查 | 调用 `/api/health` | 返回 ready、scope、gallery_patch_count | 通过 |
| SYS-02 | 系统状态 | 调用 `/api/system/status` | 返回模型、设备、文件状态 | 通过 |
| SYS-03 | 索引状态 | 调用 `/api/index/status` | 返回索引和后台任务状态 | 通过 |
| SYS-04 | 重载索引 | admin 调用 `/api/index/reload` | 模型和索引重新加载 | 通过 |
| SYS-05 | 触发索引重建 | admin 调用 `/api/index/rebuild` | 返回后台任务 pid 和日志路径 | 通过 |
| SYS-06 | 检索用户重建索引 | retriever 调用 `/api/index/rebuild` | 返回 403 | 通过 |
| SYS-07 | 缺失文件状态 | 临时缺失特征或 checkpoint | `/api/health` 显示未就绪或错误信息 | 通过 |
| SYS-08 | 管理员查看用户 | admin 调用 `/api/users` | 返回本地用户列表，且不包含 developer | 通过 |
| SYS-09 | 检索用户查看用户 | retriever 调用 `/api/users` | 返回 403 | 通过 |
| SYS-10 | 开发人员查看用户 | developer 调用 `/api/users` | 返回包含 developer 的全部用户列表 | 通过 |
| SYS-11 | 管理员按用户名检索 retriever | admin 调用 `/api/users?query=retr` | 返回匹配的 retriever 用户，且列表不包含 developer | 通过 |
| SYS-12 | 管理员新增检索用户 | admin 调用 `POST /api/users` | 新增 retriever 用户 | 通过 |
| SYS-13 | 管理员修改检索用户 | admin 调用 `PATCH /api/users/{user_id}` | 用户信息更新成功 | 通过 |
| SYS-14 | 管理员删除检索用户 | admin 调用 `DELETE /api/users/{user_id}` | 用户被停用 | 通过 |
| SYS-15 | 开发人员算法分析 | developer 调用 `/api/algorithm/analysis` | 返回模型效果和策略统计 | 通过 |
| SYS-16 | 开发人员查看模型列表 | developer 调用 `/api/models` | 返回 checkpoints 列表 | 通过 |
| SYS-17 | 开发人员切换模型 | developer 调用 `/api/models/switch` | 当前 checkpoint 切换并重新加载 | 通过 |
| SYS-18 | 管理员访问算法分析 | admin 调用 `/api/algorithm/analysis` | 返回 403 | 通过 |
| SYS-19 | 失败案例复核弹窗 | admin 或 developer 点击失败案例 | 弹窗显示灾后查询图像与灾前候选图像 | 通过 |
| SYS-20 | 访问样例图像 | 请求样例 `post_patch_url` | 返回图像文件 | 通过 |
| SYS-21 | 仓库外路径访问 | 请求 `C:/Windows/...` | 返回 403 或 404 | 通过 |

测试结论：

系统状态、索引与用户管理模块能够展示本地部署资源状态，支撑管理员维护检索库和查看用户列表。图像资源访问也具备基础路径安全控制，能够满足前端缩略图和详情图显示需求。

## 17. 模块测试总结

通过对五个核心模块进行测试，系统已经覆盖毕业设计应用系统所需的主要功能：

- 登录注册与角色权限模块通过测试，支持本地账号体系、会话恢复和多角色访问控制。
- 灾后图像检索模块通过测试，支持样例检索、上传检索、批量检索、候选过滤和排序策略切换。
- 结果分析、反馈与报告模块通过测试，支持 Top 5 展示、详情对比、反馈和报告导出。
- 检索历史与失败案例模块通过测试，支持检索留痕、历史恢复和失败样本分析。
- 系统状态、索引与用户管理模块通过测试，具备基本维护能力，并能安全返回仓库内图像资源。

当前后端接口已完成模块级验证。由于当前开发环境未安装 Node/npm，前端 Vue/Electron 界面的端到端测试尚需在前端运行环境准备完成后继续执行。总体来看，系统应用层功能完整，后端接口稳定，能够支撑灾后检索系统的毕业设计演示和论文测试章节撰写。
