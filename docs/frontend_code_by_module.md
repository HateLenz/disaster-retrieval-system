# 前端模块代码梳理

本文档按系统业务模块梳理前端实现位置。当前前端采用 Vue 3 单文件组件实现，核心业务代码集中在 `frontend/src/App.vue`，入口文件为 `frontend/src/main.js`，桌面端壳层为 `frontend/electron/main.cjs`，样式集中在 `frontend/src/styles.css`。

## 1. 前端入口与公共能力

### 涉及文件

| 文件 | 作用 |
| --- | --- |
| `frontend/src/main.js` | 创建 Vue 应用并挂载 `App.vue` |
| `frontend/src/App.vue` | 单页工作台，包含全部业务模块状态、请求方法和模板 |
| `frontend/src/styles.css` | 全局页面、表单、表格、卡片、弹窗和按钮样式 |
| `frontend/electron/main.cjs` | Electron 主进程，负责启动窗口和可选拉起后端 |

### 公共状态与工具函数

| 代码位置 | 内容 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:35` | `API_BASE` | 后端 API 基地址，默认 `http://127.0.0.1:8000` |
| `frontend/src/App.vue:87-96` | `notice`、`systemFeedback`、`loading`、`bootLoading`、`error` | 全局提示、系统操作反馈、加载状态和错误状态 |
| `frontend/src/App.vue:152-160` | `apiUrl()`、`assetUrl()` | 拼接 API 地址和图片资源地址 |
| `frontend/src/App.vue:217-234` | `downloadFile()` | 下载 Markdown 报告 |
| `frontend/src/App.vue:236-250` | `authHeaders()`、`requestJson()` | 统一带 token 的 JSON 请求 |
| `frontend/src/App.vue:252-257` | `requestPublicJson()` | 未登录也可访问的公共请求 |
| `frontend/src/App.vue:527-539` | `downloadPackage()` | 下载 zip 报告包 |
| `frontend/src/App.vue:877-887` | `watch(activeTab)` | 标签页切换时懒加载系统、算法、失败案例和用户数据 |
| `frontend/src/App.vue:890-1477` | `<template>` | 统一渲染登录页、主工作台、各模块页面和弹窗 |

### 桌面端启动代码

| 代码位置 | 内容 | 功能 |
| --- | --- | --- |
| `frontend/electron/main.cjs:7-30` | `startBackend()` | 根据环境变量决定是否拉起 FastAPI 后端 |
| `frontend/electron/main.cjs:33-54` | `createWindow()` | 创建 Electron 窗口，开发环境加载 Vite，生产环境加载构建产物 |
| `frontend/electron/main.cjs:56-75` | 生命周期事件 | 应用就绪、窗口关闭、退出时清理后端进程 |

## 2. 用户登录与角色识别模块

### 涉及前端状态

| 代码位置 | 状态 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:37` | `token` | 从 `localStorage` 恢复或保存会话 token |
| `frontend/src/App.vue:38` | `user` | 当前登录用户信息 |
| `frontend/src/App.vue:39` | `authMode` | 登录/注册模式切换 |
| `frontend/src/App.vue:40-44` | `authForm` | 用户名、密码和注册角色表单 |
| `frontend/src/App.vue:98-105` | `roleName` | 根据角色计算前端展示名称 |
| `frontend/src/App.vue:107-113` | `ready` 与权限计算属性 | 判断服务状态和角色能力 |
| `frontend/src/App.vue:124-132` | `tabs` | 按角色动态生成顶部导航 |

### 涉及前端方法

| 代码位置 | 方法 | 调用接口 | 功能 |
| --- | --- | --- | --- |
| `frontend/src/App.vue:259-278` | `loginOrRegister()` | `POST /api/auth/login`、`POST /api/auth/register` | 登录或注册，成功后写入 token 并初始化工作台 |
| `frontend/src/App.vue:280-289` | `requestPublicPost()` | 公共 POST 请求 | 登录注册阶段不带 Bearer token 的请求封装 |
| `frontend/src/App.vue:291-306` | `restoreSession()` | `GET /api/auth/me` | 页面刷新后恢复登录状态 |
| `frontend/src/App.vue:308-321` | `logout()` | `POST /api/auth/logout` | 退出登录并清空前端状态 |
| `frontend/src/App.vue:323-329` | `refreshHealthPublic()` | `GET /api/health` | 登录前检查服务状态 |
| `frontend/src/App.vue:331-337` | `refreshHealth()` | `GET /api/health` | 登录后刷新服务状态 |
| `frontend/src/App.vue:358-378` | `bootstrapAfterAuth()` | 多接口 | 登录后加载健康状态、筛选项、样例、历史和权限相关数据 |

### 涉及模板区域

| 代码位置 | 区域 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:891-948` | 登录/注册页 | 未登录时展示认证面板、角色选择、默认账号和错误提示 |
| `frontend/src/App.vue:951-983` | 主工作台头部 | 登录后展示导航、服务状态、刷新和退出按钮 |

## 3. 灾后图像检索模块

### 涉及前端状态

| 代码位置 | 状态 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:48-49` | `options`、`samples` | 灾害类型/地区筛选项和 hold 样例列表 |
| `frontend/src/App.vue:69-72` | `selectedType`、`selectedDisaster`、`filterMode`、`aggregation` | 检索筛选条件与聚合策略 |
| `frontend/src/App.vue:73-80` | `selectedFiles`、`selectedPreview`、`activeSample`、`results`、`batchItems`、`queryInfo`、`galleryInfo`、`elapsedMs` | 图像输入、预览、结果、批量状态和检索统计 |
| `frontend/src/App.vue:140-150` | `disastersForType`、`selectedCount` | 根据灾害类型联动地区列表和候选数量 |

### 涉及前端方法

| 代码位置 | 方法 | 调用接口 | 功能 |
| --- | --- | --- | --- |
| `frontend/src/App.vue:339-345` | `loadOptions()` | `GET /api/options` | 加载灾害类型、地区和候选统计 |
| `frontend/src/App.vue:347-356` | `loadSamples()` | `GET /api/samples` | 按筛选条件加载 hold 样例 |
| `frontend/src/App.vue:380-391` | `onFileChange()` | 无 | 处理上传文件和本地预览 |
| `frontend/src/App.vue:393-402` | `chooseSample()` | 无 | 选择样例并填充检索条件 |
| `frontend/src/App.vue:404-452` | `runSearch()` | `POST /api/search`、`POST /api/search-by-path`、`POST /api/batch-search` | 根据输入类型执行单图、样例或批量检索 |
| `frontend/src/App.vue:454-460` | `appendSearchFields()` | 无 | 将筛选参数写入 `FormData` |
| `frontend/src/App.vue:462-468` | `applySearchPayload()` | 无 | 将后端检索结果同步到前端状态 |

### 涉及模板区域

| 代码位置 | 区域 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:995-1089` | 检索控制区 | 灾害类型、地区、过滤模式、聚合方式、上传、样例和查询预览 |
| `frontend/src/App.vue:1092-1096` | 批量结果条 | 展示批量检索每个文件的成功或失败状态 |
| `frontend/src/App.vue:1098-1122` | Top 5 结果列表 | 展示灾前候选图像、分数、灾害和建筑元信息 |

## 4. 结果分析、反馈与报告模块

### 涉及前端状态

| 代码位置 | 状态 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:81` | `selectedResult` | 当前打开的检索结果详情 |
| `frontend/src/App.vue:83` | `feedbackNote` | 结果反馈备注 |
| `frontend/src/App.vue:90` | `downloading` | 报告或压缩包下载状态 |

### 涉及前端方法

| 代码位置 | 方法 | 调用接口 | 功能 |
| --- | --- | --- | --- |
| `frontend/src/App.vue:487-493` | `openResultDetail()` | `GET /api/results/{result_id}` | 打开结果详情，缺少完整信息时请求后端 |
| `frontend/src/App.vue:495-510` | `submitFeedback()` | `POST /api/feedback` | 提交 `correct`、`wrong` 或 `uncertain` 反馈 |
| `frontend/src/App.vue:512-525` | `downloadReport()` | `GET /api/history/{search_id}/report.md` | 下载单次检索 Markdown 报告 |

### 涉及模板区域

| 代码位置 | 区域 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:1125-1151` | 结果详情面板 | 展示查询灾后图像、命中灾前图像、元信息和反馈按钮 |
| `frontend/src/App.vue:1186-1191` | 历史详情报告入口 | 在历史详情中导出 Markdown 报告 |

## 5. 检索历史与失败案例模块

### 涉及前端状态

| 代码位置 | 状态 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:50-51` | `historyRows`、`failures` | 历史记录列表和失败案例列表 |
| `frontend/src/App.vue:56-57` | `selectedHistoryIds`、`selectedFailureIds` | 历史和失败案例批量导出勾选项 |
| `frontend/src/App.vue:82` | `selectedHistory` | 当前打开的历史详情 |
| `frontend/src/App.vue:84-86` | `failureReview`、`failureReviewNote`、`failureReviewFeedback` | 失败案例复核弹窗状态 |
| `frontend/src/App.vue:114-115` | `allHistorySelected`、`allFailuresSelected` | 全选状态 |

### 涉及前端方法

| 代码位置 | 方法 | 调用接口 | 功能 |
| --- | --- | --- | --- |
| `frontend/src/App.vue:470-475` | `loadHistory()` | `GET /api/history` | 加载历史记录 |
| `frontend/src/App.vue:477-485` | `openHistory()` | `GET /api/history/{id}` | 打开一次历史详情，并恢复结果区 |
| `frontend/src/App.vue:542-564` | 选择相关方法 | 无 | 单选、全选历史或失败案例 |
| `frontend/src/App.vue:566-570` | `exportHistory()` | `GET /api/history/export` | 导出历史 Markdown 报告包 |
| `frontend/src/App.vue:572-576` | `exportFailures()` | `GET /api/failures/export` | 导出失败案例 Markdown 报告包 |
| `frontend/src/App.vue:578-583` | `loadFailures()` | `GET /api/failures` | 加载失败案例列表 |
| `frontend/src/App.vue:802-834` | `openFailureReview()` | `GET /api/results/{id}` 或 `GET /api/history/{id}` | 打开失败案例复核弹窗 |
| `frontend/src/App.vue:836-840` | `closeFailureReview()` | 无 | 关闭失败案例复核弹窗 |
| `frontend/src/App.vue:842-865` | `submitFailureFeedback()` | `POST /api/feedback` | 在失败案例复核中提交反馈 |

### 涉及模板区域

| 代码位置 | 区域 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:1154-1192` | 历史页面 | 展示历史列表、勾选、导出、刷新和历史详情 |
| `frontend/src/App.vue:1194-1229` | 失败案例页面 | 展示失败案例列表、批量导出、刷新和复核入口 |
| `frontend/src/App.vue:1422-1475` | 失败案例复核弹窗 | 展示查询图像、命中灾前图像、反馈类型和备注 |

## 6. 系统状态、索引与用户管理模块

### 涉及前端状态

| 代码位置 | 状态 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:52-53` | `usersList`、`systemStatus` | 用户列表和系统状态 |
| `frontend/src/App.vue:55` | `savingUserId` | 用户保存/删除中的行 ID |
| `frontend/src/App.vue:58-64` | `userSearch`、`activeUserQuery`、`userForm` | 用户搜索和新增用户表单 |
| `frontend/src/App.vue:88-92` | `systemFeedback`、`userSearchStatus`、`downloading`、`savingFailureReview`、`systemAction` | 系统操作反馈和后台动作状态 |
| `frontend/src/App.vue:116-117` | `indexJob`、`indexBusy` | 索引后台任务状态 |

### 涉及前端方法

| 代码位置 | 方法 | 调用接口 | 功能 |
| --- | --- | --- | --- |
| `frontend/src/App.vue:585-601` | `loadSystemStatus()` | `GET /api/system/status` | 加载模型、索引、设备和文件状态 |
| `frontend/src/App.vue:603-625` | `reloadIndex()` | `POST /api/index/reload` | 重载模型和索引 |
| `frontend/src/App.vue:627-648` | `rebuildIndex()` | `POST /api/index/rebuild` | 触发后台重建 hold 索引 |
| `frontend/src/App.vue:699-707` | `loadUsers()` | `GET /api/users` | 加载用户列表，支持用户名查询 |
| `frontend/src/App.vue:709-727` | `searchUsers()` | `GET /api/users?query=...` | 搜索用户 |
| `frontend/src/App.vue:729-736` | `refreshUsers()` | `GET /api/users` | 刷新用户列表 |
| `frontend/src/App.vue:738-740` | `canEditUser()` | 无 | 判断当前用户是否可编辑目标用户 |
| `frontend/src/App.vue:742-758` | `createRetrieverUser()` | `POST /api/users` | 管理员新增检索用户 |
| `frontend/src/App.vue:760-784` | `saveUser()` | `PATCH /api/users/{id}` | 管理员修改检索用户 |
| `frontend/src/App.vue:786-800` | `deleteUser()` | `DELETE /api/users/{id}` | 管理员删除或停用检索用户 |

### 涉及模板区域

| 代码位置 | 区域 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:1231-1316` | 系统状态页 | 展示服务状态、索引任务、文件状态、重载和重建按钮 |
| `frontend/src/App.vue:1369-1418` | 用户管理页 | 用户搜索、新增、修改和删除 |

## 7. 算法分析与模型切换模块

算法分析在系统设计中属于算法开发人员能力，前端单独作为 `algorithm` 标签页实现；模型切换同时出现在系统状态页和算法分析页。

### 涉及前端状态

| 代码位置 | 状态 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:54` | `algorithmAnalysis` | 算法统计、失败样本和策略统计 |
| `frontend/src/App.vue:65-67` | `modelOptions`、`selectedModelPath`、`preferredSystemModelName` | 本地 checkpoint 列表、当前选择和推荐模型名 |
| `frontend/src/App.vue:118-123` | `preferredSystemModel`、`currentModelName` | 推荐模型和当前模型展示 |

### 涉及前端方法

| 代码位置 | 方法 | 调用接口 | 功能 |
| --- | --- | --- | --- |
| `frontend/src/App.vue:650-653` | `loadAlgorithmAnalysis()` | `GET /api/algorithm/analysis` | 加载检索统计、反馈统计和近期失败样本 |
| `frontend/src/App.vue:655-661` | `loadModels()` | `GET /api/models` | 加载本地 checkpoint 列表 |
| `frontend/src/App.vue:663-691` | `switchModel()` | `POST /api/models/switch` | 切换当前检索模型并刷新状态 |
| `frontend/src/App.vue:693-697` | `switchToPreferredSystemModel()` | `POST /api/models/switch` | 一键切换到推荐模型 |

### 涉及模板区域

| 代码位置 | 区域 | 功能 |
| --- | --- | --- |
| `frontend/src/App.vue:1260-1283` | 系统页模型切换面板 | developer 在系统状态页切换模型 |
| `frontend/src/App.vue:1318-1367` | 算法分析页 | 展示统计卡片、模型切换、策略统计和近期失败样本 |

## 8. 模块与 API 对应汇总

| 模块 | 前端入口 | 主要前端方法 | 后端接口 |
| --- | --- | --- | --- |
| 用户登录与角色识别 | 登录页、顶部导航 | `loginOrRegister()`、`restoreSession()`、`logout()`、`bootstrapAfterAuth()` | `/api/auth/login`、`/api/auth/register`、`/api/auth/me`、`/api/auth/logout`、`/api/health` |
| 灾后图像检索 | `retrieve` 标签页 | `loadOptions()`、`loadSamples()`、`runSearch()`、`chooseSample()`、`applySearchPayload()` | `/api/options`、`/api/samples`、`/api/search`、`/api/search-by-path`、`/api/batch-search` |
| 结果分析、反馈与报告 | 结果详情面板、历史详情 | `openResultDetail()`、`submitFeedback()`、`downloadReport()` | `/api/results/{id}`、`/api/feedback`、`/api/history/{id}/report.md` |
| 检索历史与失败案例 | `history`、`failures` 标签页 | `loadHistory()`、`openHistory()`、`loadFailures()`、`openFailureReview()`、`exportHistory()`、`exportFailures()` | `/api/history`、`/api/history/{id}`、`/api/failures`、`/api/history/export`、`/api/failures/export` |
| 系统状态、索引与用户管理 | `system`、`users` 标签页 | `loadSystemStatus()`、`reloadIndex()`、`rebuildIndex()`、`loadUsers()`、`createRetrieverUser()`、`saveUser()`、`deleteUser()` | `/api/system/status`、`/api/index/reload`、`/api/index/rebuild`、`/api/users` |
| 算法分析与模型切换 | `algorithm` 标签页、系统页模型面板 | `loadAlgorithmAnalysis()`、`loadModels()`、`switchModel()`、`switchToPreferredSystemModel()` | `/api/algorithm/analysis`、`/api/models`、`/api/models/switch` |

## 9. 代码结构特点

1. 当前前端没有拆分多个 Vue 组件，所有模块都在 `App.vue` 中通过 `activeTab` 条件渲染。
2. 角色权限主要由 `canViewFailures`、`canViewUsers`、`canManageRetrievers`、`canOperateIndex`、`canDevelop` 等计算属性控制。
3. 网络请求统一经过 `requestJson()`，除登录注册和健康检查外，默认携带 `Authorization: Bearer <token>`。
4. 图像资源不直接读取本地路径，而是使用后端返回的 `/api/assets?path=...` 地址，并通过 `assetUrl()` 转为完整 URL。
5. Electron 端只负责窗口和后端进程管理，具体业务交互仍由 Vue 前端和 FastAPI 后端完成。

## 10. 论文可引用的前端核心代码与难点代码

本节整理可写入论文“系统实现”章节的前端核心代码。代码片段均来自 `frontend/src/App.vue`，重点体现权限控制、统一请求封装、检索流程分流、结果状态回填、报告导出、失败案例复核、系统维护和模型切换等实现难点。

### 10.1 角色权限与动态导航生成

该部分用于说明系统如何根据用户角色动态生成工作台入口。前端并不是固定展示全部功能，而是通过计算属性控制失败案例、系统状态、算法分析和用户管理等页面的可见性。

```javascript
const ready = computed(() => Boolean(health.value?.ready));
const canViewFailures = computed(() => ["admin", "developer"].includes(user.value?.role));
const canViewUsers = computed(() => ["admin", "developer"].includes(user.value?.role));
const canManageRetrievers = computed(() => user.value?.role === "admin");
const canOperateIndex = computed(() => ["admin", "developer"].includes(user.value?.role));
const canDevelop = computed(() => user.value?.role === "developer");
const showSystem = computed(() => canOperateIndex.value);

const tabs = computed(() => {
  const values = [{ id: "retrieve", label: "检索", icon: Search }];
  values.push({ id: "history", label: "历史", icon: History });
  if (canViewFailures.value) values.push({ id: "failures", label: "失败案例", icon: ClipboardCheck });
  if (showSystem.value) values.push({ id: "system", label: "系统状态", icon: Settings });
  if (canDevelop.value) values.push({ id: "algorithm", label: "算法分析", icon: BarChart3 });
  if (canViewUsers.value) values.push({ id: "users", label: "用户管理", icon: Users });
  return values;
});
```

论文说明要点：

- `retriever` 只显示检索和历史。
- `admin` 增加失败案例、系统状态和用户管理。
- `developer` 增加算法分析和模型切换能力。
- 前端权限控制用于界面展示，后端接口仍负责最终鉴权。

### 10.2 统一鉴权请求封装

系统中除登录注册和健康检查外，大部分业务接口都需要 Bearer token。前端将 token 注入逻辑封装到 `requestJson()` 中，避免每个接口重复处理鉴权请求和错误解析。

```javascript
function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (token.value) headers.Authorization = `Bearer ${token.value}`;
  return headers;
}

async function requestJson(path, init = {}) {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: authHeaders(init.headers || {})
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}
```

论文说明要点：

- `authHeaders()` 统一把本地 token 写入请求头。
- `requestJson()` 统一解析 JSON 响应和错误信息。
- 该封装降低了检索、历史、反馈、系统管理等模块的重复代码量。

### 10.3 登录注册与工作台初始化

用户登录或注册成功后，前端保存 token，并调用 `bootstrapAfterAuth()` 一次性加载工作台所需的基础数据。该流程把登录态恢复、业务筛选项、样例、历史、失败案例、系统状态和用户数据串联起来。

```javascript
async function loginOrRegister() {
  error.value = "";
  notice.value = "";
  loading.value = true;
  try {
    const path = authMode.value === "login" ? "/api/auth/login" : "/api/auth/register";
    const payload = authMode.value === "login"
      ? { username: authForm.value.username, password: authForm.value.password }
      : authForm.value;
    const response = await requestPublicPost(path, payload);
    token.value = response.token;
    user.value = response.user;
    localStorage.setItem("dfr_token", token.value);
    await bootstrapAfterAuth();
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
}

async function bootstrapAfterAuth() {
  bootLoading.value = true;
  error.value = "";
  notice.value = "";
  await refreshHealth();
  try {
    await loadOptions();
    if (disastersForType.value.length && !selectedDisaster.value) {
      selectedDisaster.value = disastersForType.value[0];
    }
    await loadSamples();
    await loadHistory();
    if (showSystem.value) await loadSystemStatus();
    if (canViewFailures.value) await loadFailures();
    if (canDevelop.value) await loadAlgorithmAnalysis();
    if (canDevelop.value) await loadModels();
    if (canViewUsers.value) await loadUsers();
  } catch (err) {
    error.value = err.message;
  } finally {
    bootLoading.value = false;
  }
}
```

论文说明要点：

- 登录成功后将 token 持久化到 `localStorage`，刷新页面后可恢复会话。
- 初始化流程根据角色加载不同数据，避免普通检索用户请求管理接口。
- `bootLoading` 与 `error` 用于控制页面加载状态和错误提示。

### 10.4 单图、批量和样例检索统一入口

检索模块的难点在于系统同时支持三类输入：多文件批量上传、单文件上传和 hold 样例路径检索。前端通过 `runSearch()` 对输入类型进行分流，并把灾害类型、灾害地区、候选过滤模式和聚合策略统一传给后端。

```javascript
async function runSearch() {
  if (!ready.value || loading.value) return;
  error.value = "";
  notice.value = "";
  loading.value = true;
  results.value = [];
  batchItems.value = [];
  selectedResult.value = null;

  try {
    let payload;
    if (selectedFiles.value.length > 1) {
      const formData = new FormData();
      selectedFiles.value.forEach((file) => formData.append("images", file));
      appendSearchFields(formData);
      payload = await requestJson("/api/batch-search", { method: "POST", body: formData });
      batchItems.value = payload.items || [];
      const firstOk = batchItems.value.find((item) => item.ok);
      if (firstOk) applySearchPayload(firstOk.result);
    } else if (selectedFiles.value.length === 1) {
      const formData = new FormData();
      formData.append("image", selectedFiles.value[0]);
      appendSearchFields(formData);
      payload = await requestJson("/api/search", { method: "POST", body: formData });
      applySearchPayload(payload);
    } else if (activeSample.value?.post_patch_path) {
      payload = await requestJson("/api/search-by-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: activeSample.value.post_patch_path,
          disaster_type: selectedType.value,
          disaster: selectedDisaster.value,
          top_k: 5,
          filter_mode: filterMode.value,
          aggregation: aggregation.value
        })
      });
      applySearchPayload(payload);
    } else {
      throw new Error("请选择查询图像");
    }
    await loadHistory();
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
}
```

论文说明要点：

- 多文件上传调用 `/api/batch-search`。
- 单文件上传调用 `/api/search`。
- 样例图像不重复上传文件，而是调用 `/api/search-by-path`。
- 检索结束后调用 `loadHistory()`，实现检索结果自动留痕。

### 10.5 检索参数封装与结果状态回填

该代码用于说明前端如何把用户选择的检索参数传给后端，以及如何把后端返回的 `results`、`query`、`gallery` 和耗时同步到界面。

```javascript
function appendSearchFields(formData) {
  formData.append("disaster_type", selectedType.value);
  formData.append("disaster", selectedDisaster.value);
  formData.append("top_k", "5");
  formData.append("filter_mode", filterMode.value);
  formData.append("aggregation", aggregation.value);
}

function applySearchPayload(payload) {
  results.value = payload.results || [];
  queryInfo.value = payload.query;
  galleryInfo.value = payload.gallery;
  elapsedMs.value = payload.elapsed_ms;
  selectedResult.value = results.value[0] || null;
}
```

论文说明要点：

- `appendSearchFields()` 保证上传检索和批量检索使用一致的筛选参数。
- `applySearchPayload()` 将后端响应映射为前端展示状态。
- 默认选中 Top 1 结果，便于用户直接查看灾前灾后对比。

### 10.6 历史恢复、结果详情和反馈提交

历史模块的难点是将一次历史检索重新恢复成当前结果展示状态。反馈模块则需要在提交后刷新结果详情和失败案例列表。

```javascript
async function openHistory(row) {
  selectedHistory.value = await requestJson(`/api/history/${row.id}`);
  results.value = selectedHistory.value.results || [];
  queryInfo.value = selectedHistory.value.query;
  galleryInfo.value = selectedHistory.value.gallery;
  elapsedMs.value = selectedHistory.value.elapsed_ms;
  selectedResult.value = results.value[0] || null;
  activeTab.value = "history";
}

async function openResultDetail(item) {
  if (!item?.search_result_id) {
    selectedResult.value = item;
    return;
  }
  selectedResult.value = await requestJson(`/api/results/${item.search_result_id}`);
}

async function submitFeedback(item, feedback, note = undefined) {
  if (!item?.search_result_id) return;
  const noteValue = note ?? feedbackNote.value;
  await requestJson("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      search_result_id: item.search_result_id,
      feedback,
      note: noteValue
    })
  });
  if (note === undefined) feedbackNote.value = "";
  await openResultDetail(item);
  if (canViewFailures.value) await loadFailures();
}
```

论文说明要点：

- `openHistory()` 将历史详情恢复到结果列表和详情面板中。
- `openResultDetail()` 支持从当前结果或历史结果中打开完整详情。
- `submitFeedback()` 提交人工反馈，并同步更新结果详情与失败案例。

### 10.7 历史与失败案例批量导出

系统支持选择多条历史或失败案例导出 Markdown 报告包。前端通过勾选 ID 生成查询参数，后端返回 zip 文件。

```javascript
function updateSelection(collection, id, checked) {
  if (checked) {
    if (!collection.value.includes(id)) collection.value = [...collection.value, id];
  } else {
    collection.value = collection.value.filter((item) => item !== id);
  }
}

function setAllHistorySelection(checked) {
  selectedHistoryIds.value = checked ? historyRows.value.map((row) => row.id) : [];
}

function setAllFailureSelection(checked) {
  selectedFailureIds.value = checked ? failures.value.map((row) => row.id) : [];
}

async function exportHistory(selectedOnly = true) {
  const ids = selectedOnly ? selectedHistoryIds.value : historyRows.value.map((row) => row.id);
  const query = ids.length ? `?ids=${ids.join(",")}` : "";
  await downloadPackage(`/api/history/export${query}`, "retrieval_history.zip");
}

async function exportFailures(selectedOnly = true) {
  const ids = selectedOnly ? selectedFailureIds.value : failures.value.map((row) => row.id);
  const query = ids.length ? `?ids=${ids.join(",")}` : "";
  await downloadPackage(`/api/failures/export${query}`, "retrieval_failures.zip");
}
```

论文说明要点：

- `updateSelection()` 统一处理历史和失败案例的勾选逻辑。
- `exportHistory()` 与 `exportFailures()` 复用同一种 ID 参数拼接方式。
- 该设计支撑批量报告导出，便于答辩展示和后续复盘。

### 10.8 失败案例复核弹窗

失败案例可能来自低分检索记录，也可能来自某个被标记为错误的具体结果。前端需要同时兼容 `search_result_id` 和 `history id` 两类入口。

```javascript
async function openFailureReview(row) {
  error.value = "";
  notice.value = "";
  try {
    if (row?.search_result_id) {
      const detail = await requestJson(`/api/results/${row.search_result_id}`);
      failureReviewNote.value = row.feedback_note || detail.feedback_note || "";
      failureReviewFeedback.value = row.feedback || detail.feedback || "wrong";
      failureReview.value = {
        row,
        item: detail,
        queryUrl: assetUrl(detail.query?.source_url),
        preUrl: assetUrl(detail.pre_patch_url || detail.pre_image_url),
        title: shortRecordTitle(row, "案例")
      };
      return;
    }

    const detail = await requestJson(`/api/history/${row.id}`);
    const firstResult = detail.results?.[0] || null;
    failureReviewNote.value = row.feedback_note || "";
    failureReviewFeedback.value = row.feedback || firstResult?.feedback || "wrong";
    failureReview.value = {
      row,
      item: firstResult,
      queryUrl: assetUrl(detail.query?.source_url),
      preUrl: assetUrl(firstResult?.pre_patch_url || firstResult?.pre_image_url),
      title: shortRecordTitle(row, "案例")
    };
  } catch (err) {
    error.value = err.message;
  }
}
```

论文说明要点：

- 对错误反馈结果，直接读取 `/api/results/{id}`。
- 对低分历史记录，读取 `/api/history/{id}` 并默认展示 Top 1。
- 通过 `assetUrl()` 把后端资源路径转为可显示图片地址。

### 10.9 系统状态、索引重载与索引重建

系统维护模块面向管理员和算法开发人员。前端通过 `systemAction` 防止重复点击，通过 `indexBusy` 判断后台索引任务是否正在运行。

```javascript
async function reloadIndex() {
  if (!canOperateIndex.value || indexBusy.value) return;
  error.value = "";
  notice.value = "";
  systemFeedback.value = null;
  systemAction.value = "reload";
  try {
    systemStatus.value = await requestJson("/api/index/reload", { method: "POST" });
    await refreshHealth();
    notice.value = systemStatus.value?.retrieval?.ready
      ? "索引和模型已重新加载。"
      : `重载完成，但服务未就绪：${systemStatus.value?.retrieval?.error || "请检查模型和索引文件。"}`;
    systemFeedback.value = {
      type: systemStatus.value?.retrieval?.ready ? "success" : "error",
      message: notice.value
    };
  } catch (err) {
    error.value = err.message;
    systemFeedback.value = { type: "error", message: err.message };
  } finally {
    systemAction.value = "";
  }
}

async function rebuildIndex() {
  if (!canOperateIndex.value || indexBusy.value) return;
  error.value = "";
  notice.value = "";
  systemFeedback.value = null;
  systemAction.value = "rebuild";
  try {
    systemStatus.value = await requestJson("/api/index/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ split: "hold", amp: true })
    });
    const job = systemStatus.value?.index_job || systemStatus.value?.job || {};
    notice.value = `索引重建任务已启动${job.pid ? `，PID ${job.pid}` : ""}。`;
    systemFeedback.value = { type: "success", message: notice.value };
  } catch (err) {
    error.value = err.message;
    systemFeedback.value = { type: "error", message: err.message };
  } finally {
    systemAction.value = "";
  }
}
```

论文说明要点：

- `reloadIndex()` 用于重新加载模型、特征和元信息。
- `rebuildIndex()` 用于触发后端后台任务重建 hold 索引。
- 前端通过状态变量展示任务反馈和异常信息，增强系统可维护性。

### 10.10 算法分析与本地模型切换

算法开发人员可以读取本地 checkpoint 列表并切换模型。切换成功后，前端刷新健康状态和模型列表，保证界面显示与后端当前模型一致。

```javascript
async function loadModels() {
  if (!canDevelop.value) return;
  const payload = await requestJson("/api/models");
  modelOptions.value = payload.models || [];
  const active = modelOptions.value.find((item) => item.active);
  selectedModelPath.value = active?.path || modelOptions.value[0]?.path || "";
}

async function switchModel() {
  if (!selectedModelPath.value || systemAction.value) return;
  error.value = "";
  notice.value = "";
  systemFeedback.value = null;
  systemAction.value = "model";
  try {
    const payload = await requestJson("/api/models/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint_path: selectedModelPath.value })
    });
    systemStatus.value = payload;
    await refreshHealth();
    await loadModels();
    notice.value = systemStatus.value?.retrieval?.ready
      ? `模型已切换为 ${currentModelName.value}。`
      : `模型切换完成，但服务未就绪：${systemStatus.value?.retrieval?.error || "请检查 checkpoint 和索引文件。"}`;
    systemFeedback.value = {
      type: systemStatus.value?.retrieval?.ready ? "success" : "error",
      message: notice.value
    };
  } catch (err) {
    error.value = err.message;
    systemFeedback.value = { type: "error", message: err.message };
  } finally {
    systemAction.value = "";
  }
}
```

论文说明要点：

- 该功能体现系统不仅能检索，还具备模型维护和算法调试能力。
- 模型切换后立即刷新 `/api/health`，避免前端展示旧状态。
- 仅 `developer` 角色可调用该功能，符合算法开发人员职责划分。

### 10.11 标签页懒加载

为减少登录后一次性请求过多接口，部分模块采用标签页切换时加载的方式。这样既保持首次进入速度，也保证系统状态、算法分析、用户列表等数据尽量实时。

```javascript
watch(activeTab, async (tab) => {
  if (tab === "history") await loadHistory().catch((err) => { error.value = err.message; });
  if (tab === "failures") await loadFailures().catch((err) => { error.value = err.message; });
  if (tab === "system") await loadSystemStatus().catch((err) => { error.value = err.message; });
  if (tab === "system") await loadModels().catch((err) => { error.value = err.message; });
  if (tab === "algorithm") await loadAlgorithmAnalysis().catch((err) => { error.value = err.message; });
  if (tab === "algorithm") await loadModels().catch((err) => { error.value = err.message; });
  if (tab === "users") await loadUsers(activeUserQuery.value).catch((err) => { error.value = err.message; });
});

onMounted(restoreSession);
```

论文说明要点：

- `watch(activeTab)` 实现按需加载。
- `onMounted(restoreSession)` 实现页面刷新后的会话恢复。
- 每个异步请求独立捕获异常，避免某个模块失败影响整个工作台。
