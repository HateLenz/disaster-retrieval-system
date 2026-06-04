<script setup>
import { computed, onMounted, ref, watch } from "vue";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ClipboardCheck,
  Database,
  Download,
  FileDown,
  FileText,
  FolderCog,
  History,
  Image as ImageIcon,
  Loader2,
  LogIn,
  LogOut,
  MapPin,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings,
  Shield,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Upload,
  X,
  UserPlus,
  Users
} from "lucide-vue-next";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const SEARCH_TOP_K = 10;

const token = ref(localStorage.getItem("dfr_token") || "");
const user = ref(null);
const authMode = ref("login");
const authForm = ref({
  username: "admin",
  password: "admin123",
  role: "retriever"
});

const activeTab = ref("retrieve");
const health = ref(null);
const options = ref({ disaster_types: [], disasters: [], disasters_by_type: {}, counts: [] });
const samples = ref([]);
const historyRows = ref([]);
const failures = ref([]);
const usersList = ref([]);
const systemStatus = ref(null);
const algorithmAnalysis = ref(null);
const savingUserId = ref(null);
const selectedHistoryIds = ref([]);
const selectedFailureIds = ref([]);
const userSearch = ref("");
const activeUserQuery = ref("");
const userForm = ref({
  username: "",
  password: "",
  role: "retriever"
});
const modelOptions = ref([]);
const selectedModelPath = ref("");
const checkpointLabels = {
  "clip_visual_baseline_smoke_stage1.pt": "检索模型1.0",
  "clip_visual_baseline_smoke_stage1_v2.pt": "检索模型1.1"
};

const selectedType = ref("");
const selectedDisaster = ref("");
const filterMode = ref("both");
const aggregation = ref("top_m");
const selectedFiles = ref([]);
const selectedPreview = ref("");
const activeSample = ref(null);
const results = ref([]);
const batchItems = ref([]);
const queryInfo = ref(null);
const galleryInfo = ref(null);
const elapsedMs = ref(null);
const selectedResult = ref(null);
const selectedHistory = ref(null);
const feedbackNote = ref("");
const failureReview = ref(null);
const failureReviewNote = ref("");
const failureReviewFeedback = ref("wrong");
const notice = ref("");
const systemFeedback = ref(null);
const userSearchStatus = ref("");
const downloading = ref(false);
const savingFailureReview = ref(false);
const systemAction = ref("");

const loading = ref(false);
const bootLoading = ref(false);
const error = ref("");

const roleName = computed(() => {
  const names = {
    retriever: "检索用户",
    admin: "系统管理员",
    developer: "算法开发人员"
  };
  return names[user.value?.role] || "-";
});

const ready = computed(() => Boolean(health.value?.ready));
const canViewFailures = computed(() => ["admin", "developer"].includes(user.value?.role));
const canViewUsers = computed(() => ["admin", "developer"].includes(user.value?.role));
const canManageRetrievers = computed(() => user.value?.role === "admin");
const canOperateIndex = computed(() => ["admin", "developer"].includes(user.value?.role));
const canDevelop = computed(() => user.value?.role === "developer");
const showSystem = computed(() => canOperateIndex.value);
const allHistorySelected = computed(() => historyRows.value.length > 0 && historyRows.value.every((row) => selectedHistoryIds.value.includes(row.id)));
const allFailuresSelected = computed(() => failures.value.length > 0 && failures.value.every((row) => selectedFailureIds.value.includes(row.id)));
const indexJob = computed(() => systemStatus.value?.index_job || systemStatus.value?.job || {});
const indexBusy = computed(() => Boolean(systemAction.value) || indexJob.value?.status === "running");
const currentModelName = computed(() => {
  const current = systemStatus.value?.retrieval?.checkpoint || health.value?.checkpoint || "";
  return current ? checkpointDisplayName(current) : "-";
});

const tabs = computed(() => {
  const values = [{ id: "retrieve", label: "检索", icon: Search }];
  values.push({ id: "history", label: "历史", icon: History });
  if (canViewFailures.value) values.push({ id: "failures", label: "失败案例", icon: ClipboardCheck });
  if (showSystem.value) values.push({ id: "system", label: "系统状态", icon: Settings });
  if (canDevelop.value) values.push({ id: "algorithm", label: "算法分析", icon: BarChart3 });
  if (canViewUsers.value) values.push({ id: "users", label: "用户管理", icon: Users });
  return values;
});

const statusText = computed(() => {
  if (!health.value) return "连接中";
  if (health.value.ready) return "在线";
  return "未就绪";
});

const disastersForType = computed(() => {
  if (selectedType.value && options.value.disasters_by_type?.[selectedType.value]) {
    return options.value.disasters_by_type[selectedType.value];
  }
  return options.value.disasters || [];
});

const selectedCount = computed(() => {
  const rows = options.value.counts || [];
  return rows.find((item) => item.disaster_type === selectedType.value && item.disaster === selectedDisaster.value) || null;
});

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function assetUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return apiUrl(path);
}

function readableQueryLabel(label, disaster = "") {
  const source = String(label || disaster || "query");
  const base = source.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || source;
  const lower = base.toLowerCase();
  const markers = ["fire", "flood", "flooding", "wind", "earthquake", "hurricane", "tsunami", "volcano"];
  let cutAt = base.length;
  for (const marker of markers) {
    const index = lower.indexOf(marker);
    if (index > 0) cutAt = Math.min(cutAt, index);
  }
  const display = base.slice(0, cutAt).replace(/[_-]+/g, " ").trim();
  return display || displayDisasterLabel(disaster) || base;
}

function shortRecordTitle(row, prefix = "记录") {
  const location = displayDisasterLabel(row?.disaster);
  const id = row?.id ? `#${row.id}` : "";
  return [location !== "-" ? location : "", `${prefix}${id}`].filter(Boolean).join(" ");
}

function originalQueryTitle(row) {
  return String(row?.query_label || row?.disaster || "").trim();
}

function displayDisasterLabel(value) {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const hiddenTokens = new Set([
    "fire",
    "wildfire",
    "bushfire",
    "flood",
    "flooding",
    "hurricane",
    "tornado",
    "wind",
    "earthquake",
    "tsunami",
    "volcano",
    "disaster"
  ]);
  const parts = raw.split(/[-_]+/).filter(Boolean);
  const visibleParts = parts.filter((part) => !hiddenTokens.has(part.toLowerCase()));
  return visibleParts.length ? visibleParts.join("-") : raw;
}

function fileNameFromResponse(response, fallback) {
  const disposition = response.headers.get("Content-Disposition") || "";
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) return decodeURIComponent(utf8Match[1]);
  const nameMatch = disposition.match(/filename="?([^"]+)"?/i);
  if (nameMatch?.[1]) return nameMatch[1];
  return fallback;
}

function checkpointFileName(value) {
  return String(value || "").split(/[\\/]/).pop();
}

function checkpointDisplayName(value) {
  const fileName = checkpointFileName(value);
  return checkpointLabels[fileName] || fileName || "-";
}

function modelOptionLabel(item) {
  const label = item.display_name || checkpointDisplayName(item.name || item.path);
  return `${item.active ? "当前：" : ""}${label}`;
}

async function downloadFile(path, fallbackName) {
  const response = await fetch(apiUrl(path), { headers: authHeaders() });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = fileNameFromResponse(response, fallbackName);
  anchor.rel = "noopener";
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

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

async function requestPublicJson(path) {
  const response = await fetch(apiUrl(path));
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

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

async function requestPublicPost(path, payload) {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

async function restoreSession() {
  if (!token.value) {
    await refreshHealthPublic();
    return;
  }
  try {
    const payload = await requestJson("/api/auth/me");
    user.value = payload.user;
    await bootstrapAfterAuth();
  } catch {
    token.value = "";
    user.value = null;
    localStorage.removeItem("dfr_token");
    await refreshHealthPublic();
  }
}

async function logout() {
  try {
    await requestJson("/api/auth/logout", { method: "POST" });
  } catch {
    // Ignore expired sessions during logout.
  }
  token.value = "";
  user.value = null;
  localStorage.removeItem("dfr_token");
  activeTab.value = "retrieve";
  failureReview.value = null;
  notice.value = "";
  userSearchStatus.value = "";
}

async function refreshHealthPublic() {
  try {
    health.value = await requestPublicJson("/api/health");
  } catch (err) {
    health.value = { ready: false, error: err.message };
  }
}

async function refreshHealth() {
  try {
    health.value = await requestJson("/api/health");
  } catch (err) {
    health.value = { ready: false, error: err.message };
  }
}

async function loadOptions() {
  if (!ready.value || !user.value) return;
  options.value = await requestJson("/api/options");
  if (!selectedType.value && options.value.disaster_types?.length) {
    selectedType.value = options.value.disaster_types[0];
  }
}

async function loadSamples() {
  if (!ready.value || !user.value) return;
  const params = new URLSearchParams();
  params.set("limit", "10");
  const payload = await requestJson(`/api/samples?${params.toString()}`);
  samples.value = payload.samples || [];
  if (!activeSample.value && samples.value.length) chooseSample(samples.value[0]);
}

async function bootstrapAfterAuth() {
  bootLoading.value = true;
  error.value = "";
  notice.value = "";
  await refreshHealth();
  try {
    await loadOptions();
    if (disastersForType.value.length && !selectedDisaster.value) selectedDisaster.value = disastersForType.value[0];
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

function onFileChange(event) {
  selectedFiles.value = Array.from(event.target.files || []);
  activeSample.value = null;
  results.value = [];
  batchItems.value = [];
  selectedResult.value = null;
  queryInfo.value = null;
  galleryInfo.value = null;
  elapsedMs.value = null;
  if (selectedPreview.value) URL.revokeObjectURL(selectedPreview.value);
  selectedPreview.value = selectedFiles.value[0] ? URL.createObjectURL(selectedFiles.value[0]) : "";
}

function chooseSample(sample) {
  activeSample.value = sample;
  selectedFiles.value = [];
  selectedPreview.value = "";
  const availableDisasters = options.value.disasters_by_type?.[sample.disaster_type] || [];
  selectedType.value = sample.disaster_type;
  selectedDisaster.value = availableDisasters.includes(sample.disaster) ? sample.disaster : "";
  if (!selectedDisaster.value && ["both", "disaster"].includes(filterMode.value)) {
    filterMode.value = "type";
  }
  results.value = [];
  batchItems.value = [];
  selectedResult.value = null;
}

function activeSampleMatchesSelection() {
  return activeSample.value?.disaster_type === selectedType.value && activeSample.value?.disaster === selectedDisaster.value;
}

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
          top_k: SEARCH_TOP_K,
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

function appendSearchFields(formData) {
  formData.append("disaster_type", selectedType.value);
  formData.append("disaster", selectedDisaster.value);
  formData.append("top_k", String(SEARCH_TOP_K));
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

async function loadHistory() {
  if (!user.value) return;
  const payload = await requestJson("/api/history?limit=80");
  historyRows.value = payload.history || [];
  selectedHistoryIds.value = selectedHistoryIds.value.filter((id) => historyRows.value.some((row) => row.id === id));
}

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

async function downloadReport(searchId) {
  error.value = "";
  notice.value = "Preparing report download...";
  downloading.value = true;
  try {
    await downloadFile(`/api/history/${searchId}/report.md`, `retrieval_report_${searchId}.md`);
    notice.value = "Markdown download started.";
  } catch (err) {
    error.value = err.message;
    notice.value = "";
  } finally {
    downloading.value = false;
  }
}

async function downloadPackage(path, fallbackName) {
  error.value = "";
  notice.value = "Preparing export package...";
  downloading.value = true;
  try {
    await downloadFile(path, fallbackName);
    notice.value = "Export package download started.";
  } catch (err) {
    error.value = err.message;
    notice.value = "";
  } finally {
    downloading.value = false;
  }
}

function updateSelection(collection, id, checked) {
  if (checked) {
    if (!collection.value.includes(id)) collection.value = [...collection.value, id];
  } else {
    collection.value = collection.value.filter((item) => item !== id);
  }
}

function updateHistorySelection(id, checked) {
  updateSelection(selectedHistoryIds, id, checked);
}

function updateFailureSelection(id, checked) {
  updateSelection(selectedFailureIds, id, checked);
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

async function loadFailures() {
  if (!canViewFailures.value) return;
  const payload = await requestJson("/api/failures?score_threshold=0.35&limit=80");
  failures.value = payload.failures || [];
  selectedFailureIds.value = selectedFailureIds.value.filter((id) => failures.value.some((row) => row.id === id));
}

async function loadSystemStatus() {
  if (!showSystem.value) return;
  error.value = "";
  notice.value = "";
  systemFeedback.value = null;
  systemAction.value = "status";
  try {
    systemStatus.value = await requestJson("/api/system/status");
    notice.value = "系统状态已刷新。";
    systemFeedback.value = { type: "success", message: "系统状态已刷新。" };
  } catch (err) {
    error.value = err.message;
    systemFeedback.value = { type: "error", message: err.message };
  } finally {
    systemAction.value = "";
  }
}

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

async function loadAlgorithmAnalysis() {
  if (!canDevelop.value) return;
  algorithmAnalysis.value = await requestJson("/api/algorithm/analysis?score_threshold=0.35&limit=10");
}

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

async function loadUsers(query = userSearch.value) {
  if (!canViewUsers.value) return;
  const cleanQuery = typeof query === "string" ? query.trim() : "";
  const params = new URLSearchParams();
  if (cleanQuery) params.set("query", cleanQuery);
  const path = params.toString() ? `/api/users?${params.toString()}` : "/api/users";
  const payload = await requestJson(path);
  usersList.value = (payload.users || []).map((item) => ({ ...item, new_password: "" }));
}

async function searchUsers(event) {
  event?.preventDefault?.();
  error.value = "";
  const cleanQuery = userSearch.value.trim();
  activeUserQuery.value = cleanQuery;
  userSearchStatus.value = "Searching users...";
  notice.value = "Searching users...";
  try {
    await loadUsers(cleanQuery);
    userSearchStatus.value = usersList.value.length
      ? `Search complete: ${usersList.value.length} user(s) matched.`
      : "No users matched the query.";
    notice.value = userSearchStatus.value;
  } catch (err) {
    error.value = err.message;
    userSearchStatus.value = "";
    notice.value = "";
  }
}

async function refreshUsers() {
  error.value = "";
  notice.value = "";
  userSearch.value = "";
  activeUserQuery.value = "";
  userSearchStatus.value = "";
  await loadUsers("");
}

function canEditUser(item) {
  return canManageRetrievers.value && item?.role === "retriever";
}

async function createRetrieverUser() {
  if (!canManageRetrievers.value) return;
  error.value = "";
  notice.value = "";
  try {
    const payload = await requestJson("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...userForm.value, role: "retriever" })
    });
    userForm.value = { username: "", password: "", role: "retriever" };
    await refreshUsers();
    notice.value = "Retriever user created.";
  } catch (err) {
    error.value = err.message;
  }
}

async function saveUser(item) {
  if (!canEditUser(item)) return;
  savingUserId.value = item.id;
  error.value = "";
  notice.value = "";
  try {
    const payload = await requestJson(`/api/users/${item.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: item.username,
        role: "retriever",
        password: item.new_password || null
      })
    });
    if (user.value?.id === item.id) user.value = payload.user;
    await loadUsers(activeUserQuery.value);
    notice.value = "User information updated.";
  } catch (err) {
    error.value = err.message;
    await loadUsers(activeUserQuery.value);
  } finally {
    savingUserId.value = null;
  }
}

async function deleteUser(item) {
  if (!canEditUser(item)) return;
  savingUserId.value = item.id;
  error.value = "";
  notice.value = "";
  try {
    await requestJson(`/api/users/${item.id}`, { method: "DELETE" });
    usersList.value = usersList.value.filter((row) => row.id !== item.id);
    notice.value = "User deleted.";
  } catch (err) {
    error.value = err.message;
  } finally {
    savingUserId.value = null;
  }
}

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

function closeFailureReview() {
  failureReview.value = null;
  failureReviewNote.value = "";
  failureReviewFeedback.value = "wrong";
}

async function submitFailureFeedback() {
  if (!failureReview.value?.item?.search_result_id) return;
  error.value = "";
  notice.value = "";
  savingFailureReview.value = true;
  try {
    await submitFeedback(
      failureReview.value.item,
      failureReviewFeedback.value,
      failureReviewNote.value
    );
    await loadFailures();
    if (canDevelop.value) await loadAlgorithmAnalysis().catch(() => {});
    await openFailureReview(failureReview.value.row);
    notice.value = "Failure-case feedback has been saved.";
  } catch (err) {
    error.value = err.message;
  } finally {
    savingFailureReview.value = false;
  }
}

watch(selectedType, async () => {
  if (!user.value) return;
  if (selectedDisaster.value && !disastersForType.value.includes(selectedDisaster.value)) selectedDisaster.value = "";
  if (!activeSampleMatchesSelection()) activeSample.value = null;
});

watch(selectedDisaster, async () => {
  if (!user.value) return;
  if (!activeSampleMatchesSelection()) activeSample.value = null;
});

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
</script>

<template>
  <main v-if="!user" class="auth-shell">
    <section class="auth-panel">
      <div>
        <div class="eyebrow">xBD Hold Offline Retrieval</div>
        <h1>灾后检索系统</h1>
        <p>本地模型、本地索引、本地账号体系。</p>
      </div>

      <div class="auth-tabs">
        <button :class="{ active: authMode === 'login' }" type="button" @click="authMode = 'login'">
          <LogIn :size="16" /> 登录
        </button>
        <button :class="{ active: authMode === 'register' }" type="button" @click="authMode = 'register'">
          <UserPlus :size="16" /> 注册
        </button>
      </div>

      <div class="form-grid">
        <label>
          <span>用户名</span>
          <input v-model="authForm.username" autocomplete="username" />
        </label>
        <label>
          <span>密码</span>
          <input v-model="authForm.password" type="password" autocomplete="current-password" />
        </label>
        <label v-if="authMode === 'register'">
          <span>角色</span>
          <select v-model="authForm.role">
            <option value="retriever">检索用户</option>
            <option value="admin">系统管理员</option>
            <option value="developer">算法开发人员</option>
          </select>
        </label>
      </div>

      <button class="primary-action" type="button" :disabled="loading" @click="loginOrRegister">
        <Loader2 v-if="loading" class="spin" :size="18" />
        <LogIn v-else :size="18" />
        <span>{{ authMode === "login" ? "登录系统" : "注册并登录" }}</span>
      </button>

      <div class="demo-accounts">
        <span>演示账号：admin/admin123、developer/developer123、retriever/retriever123</span>
      </div>

      <div class="status-card">
        <span :class="{ online: ready }"><CheckCircle2 :size="16" /> {{ statusText }}</span>
        <small>{{ health?.scope || "未加载索引" }} / {{ health?.gallery_patch_count || 0 }} patches</small>
      </div>

      <div v-if="error || health?.error" class="error-banner">
        <AlertCircle :size="18" />
        <span>{{ error || health?.error }}</span>
      </div>
    </section>
  </main>

  <main v-else class="app-shell">
    <header class="topbar">
      <div>
        <div class="eyebrow">xBD Hold</div>
        <h1>灾后检索系统</h1>
      </div>
      <nav class="tabbar">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          :class="{ active: activeTab === tab.id }"
          type="button"
          @click="activeTab = tab.id"
        >
          <component :is="tab.icon" :size="16" />
          {{ tab.label }}
        </button>
      </nav>
      <div class="status-cluster">
        <div class="status-pill" :class="{ online: ready }">
          <CheckCircle2 v-if="ready" :size="16" />
          <AlertCircle v-else :size="16" />
          <span>{{ statusText }}</span>
        </div>
        <div class="metric">
          <Shield :size="17" />
          <span>{{ user.username }} / {{ roleName }}</span>
        </div>
        <button class="icon-button" type="button" title="刷新" @click="bootstrapAfterAuth">
          <RefreshCw :size="17" />
        </button>
        <button class="icon-button" type="button" title="退出" @click="logout">
          <LogOut :size="17" />
        </button>
      </div>
    </header>

    <div v-if="notice" class="info-banner">
      <CheckCircle2 :size="18" />
      <span>{{ notice }}</span>
    </div>
    <div v-if="error && activeTab !== 'retrieve' && !(activeTab === 'system' && systemFeedback)" class="error-banner page-error-banner">
      <AlertCircle :size="18" />
      <span>{{ error }}</span>
    </div>

    <section v-if="activeTab === 'retrieve'" class="workspace">
      <aside class="query-panel">
        <div class="field-row">
          <label>
            <span>灾害类型</span>
            <select v-model="selectedType" :disabled="!ready">
              <option v-for="item in options.disaster_types" :key="item" :value="item">{{ item }}</option>
            </select>
          </label>
          <label>
            <span>灾害地区</span>
            <select v-model="selectedDisaster" :disabled="!ready">
              <option value="">全部地区</option>
              <option v-for="item in disastersForType" :key="item" :value="item">{{ displayDisasterLabel(item) }}</option>
            </select>
          </label>
          <label>
            <span>候选过滤</span>
            <select v-model="filterMode">
              <option value="both">类型 + 地区</option>
              <option value="type">仅灾害类型</option>
              <option value="disaster">仅灾害地区</option>
              <option value="none">全库检索</option>
            </select>
          </label>
          <label>
            <span>排序策略</span>
            <select v-model="aggregation">
              <option value="top_m">Top-M 均值</option>
              <option value="max">最高分</option>
              <option value="mean">均值</option>
              <option value="sum">求和</option>
              <option value="vote">投票数</option>
            </select>
          </label>
        </div>

        <div class="count-strip">
          <div><strong>{{ selectedCount?.tiles || 0 }}</strong><span>tiles</span></div>
          <div><strong>{{ selectedCount?.patches || 0 }}</strong><span>patches</span></div>
          <div><strong>{{ health?.scope || "-" }}</strong><span>scope</span></div>
        </div>

        <label class="drop-zone" :class="{ disabled: !ready }">
          <input type="file" multiple accept="image/*,.tif,.tiff" :disabled="!ready" @change="onFileChange" />
          <Upload :size="26" />
          <span>{{ selectedFiles.length ? `${selectedFiles.length} 个文件` : "选择灾后图像，可多选批量检索" }}</span>
        </label>

        <div v-if="selectedPreview || activeSample" class="preview-frame">
          <img :src="selectedPreview || assetUrl(activeSample?.post_patch_url)" alt="query preview" />
        </div>

        <button class="primary-action" type="button" :disabled="!ready || loading || (!selectedFiles.length && !activeSample)" @click="runSearch">
          <Loader2 v-if="loading" class="spin" :size="18" />
          <Search v-else :size="18" />
          <span>{{ selectedFiles.length > 1 ? "批量检索" : `检索 Top ${SEARCH_TOP_K}` }}</span>
        </button>

        <div class="sample-list">
          <div class="panel-title"><ImageIcon :size="17" /><span>Hold 样例</span></div>
          <button
            v-for="sample in samples"
            :key="sample.positive_id"
            class="sample-item"
            :class="{ active: activeSample?.positive_id === sample.positive_id }"
            type="button"
            @click="chooseSample(sample)"
          >
            <img :src="assetUrl(sample.post_patch_url)" alt="" />
            <span>
              <strong>{{ displayDisasterLabel(sample.disaster) }}</strong>
              <small>{{ sample.disaster_type }}</small>
            </span>
          </button>
        </div>
      </aside>

      <section class="results-panel">
        <div class="result-toolbar">
          <div>
            <h2>Top {{ SEARCH_TOP_K }} 灾前结果</h2>
            <p v-if="queryInfo">
              {{ queryInfo.patch_count }} query patches / {{ galleryInfo?.candidate_tile_count || 0 }} candidate tiles / {{ elapsedMs }} ms
            </p>
            <p v-else>{{ bootLoading ? "加载索引状态" : "等待查询" }}</p>
          </div>
          <div class="toolbar-badges">
            <span><Activity :size="15" /> {{ filterMode }}</span>
            <span><BarChart3 :size="15" /> {{ aggregation }}</span>
            <span><MapPin :size="15" /> {{ displayDisasterLabel(selectedDisaster) }}</span>
          </div>
        </div>

        <div v-if="error || health?.error" class="error-banner">
          <AlertCircle :size="18" />
          <span>{{ error || health?.error }}</span>
        </div>

        <div v-if="batchItems.length" class="batch-strip">
          <button v-for="item in batchItems" :key="item.filename" type="button" :class="{ failed: !item.ok }" @click="item.ok && applySearchPayload(item.result)">
            {{ item.ok ? "完成" : "失败" }}：{{ item.filename }}
          </button>
        </div>

        <div v-if="!results.length" class="empty-state">
          <ImageIcon :size="42" />
          <span>{{ loading ? "检索中" : "暂无结果" }}</span>
        </div>

        <div v-else class="result-layout">
          <div class="result-grid">
            <article
              v-for="item in results"
              :key="`${item.rank}-${item.positive_id}`"
              class="result-card"
              :class="{ selected: selectedResult?.positive_id === item.positive_id }"
              @click="openResultDetail(item)"
            >
              <div class="rank">#{{ item.rank }}</div>
              <img class="result-image" :src="assetUrl(item.pre_patch_url || item.pre_image_url)" alt="" />
              <div class="result-body">
                <div class="score-line"><strong>{{ item.score }}</strong><span>best {{ item.best_patch_score }}</span></div>
                <h3>{{ item.tile_id }}</h3>
                <dl>
                  <div><dt>damage</dt><dd>{{ item.damage_label }}</dd></div>
                  <div><dt>hits</dt><dd>{{ item.patch_hit_count }}</dd></div>
                  <div><dt>patches</dt><dd>{{ item.tile_patch_count }}</dd></div>
                </dl>
              </div>
            </article>
          </div>
          <aside v-if="selectedResult" class="detail-panel">
            <h3>结果详情与前后对比</h3>
            <div class="compare-grid">
              <figure>
                <img :src="assetUrl(queryInfo?.source_url || activeSample?.post_patch_url)" alt="" />
                <figcaption>查询灾后图像</figcaption>
              </figure>
              <figure>
                <img :src="assetUrl(selectedResult.pre_patch_url)" alt="" />
                <figcaption>命中灾前图像</figcaption>
              </figure>
            </div>
            <div class="meta-list">
              <span>tile：{{ selectedResult.tile_id }}</span>
              <span>building：{{ selectedResult.building_uid }}</span>
              <span>score：{{ selectedResult.score }}</span>
              <span>damage：{{ selectedResult.damage_label }}</span>
            </div>
            <textarea v-model="feedbackNote" placeholder="反馈备注"></textarea>
            <div class="button-row">
              <button type="button" @click="submitFeedback(selectedResult, 'correct')"><ThumbsUp :size="15" /> 正确</button>
              <button type="button" @click="submitFeedback(selectedResult, 'wrong')"><ThumbsDown :size="15" /> 错误</button>
              <button type="button" @click="submitFeedback(selectedResult, 'uncertain')">不确定</button>
            </div>
          </aside>
        </div>
      </section>
    </section>

    <section v-if="activeTab === 'history'" class="page-panel">
      <div class="page-header">
        <div><h2>检索历史</h2><p>检索用户仅查看自己的记录，管理员和算法开发人员可查看全部记录。</p></div>
        <div class="button-row">
          <label class="check-action">
            <input type="checkbox" :checked="allHistorySelected" @change="setAllHistorySelection($event.target.checked)" />
            全选
          </label>
          <button class="secondary-action" type="button" :disabled="!selectedHistoryIds.length" @click="exportHistory(true)">
            <FileDown :size="16" /> 导出选中 MD 包
          </button>
          <button class="secondary-action" type="button" :disabled="!historyRows.length" @click="exportHistory(false)">
            <Download :size="16" /> 导出全部 MD 包
          </button>
          <button class="secondary-action" type="button" @click="loadHistory"><RefreshCw :size="16" /> 刷新</button>
        </div>
      </div>
      <div class="table-list">
        <div v-for="row in historyRows" :key="row.id" class="table-row history-row" @click="openHistory(row)">
          <input
            type="checkbox"
            :checked="selectedHistoryIds.includes(row.id)"
            @click.stop
            @change="updateHistorySelection(row.id, $event.target.checked)"
          />
          <span>#{{ row.id }}</span>
          <strong :title="originalQueryTitle(row)">{{ shortRecordTitle(row) }}</strong>
          <span>{{ row.disaster_type }} / {{ displayDisasterLabel(row.disaster) }}</span>
          <span>{{ row.top_score ?? "-" }}</span>
          <small>{{ row.query_time }}</small>
        </div>
      </div>
      <div v-if="selectedHistory" class="history-detail">
        <h3>记录 #{{ selectedHistory.id }}</h3>
        <button class="secondary-action" type="button" @click="downloadReport(selectedHistory.id)">
          <Download :size="16" /> 下载 Markdown
        </button>
      </div>
    </section>

    <section v-if="activeTab === 'failures'" class="page-panel">
      <div class="page-header">
        <div><h2>失败案例库</h2><p>包含低分检索记录和被标记为错误的结果。</p></div>
        <div class="button-row">
          <label class="check-action">
            <input type="checkbox" :checked="allFailuresSelected" @change="setAllFailureSelection($event.target.checked)" />
            全选
          </label>
          <button class="secondary-action" type="button" :disabled="!selectedFailureIds.length" @click="exportFailures(true)">
            <FileDown :size="16" /> 导出选中 MD 包
          </button>
          <button class="secondary-action" type="button" :disabled="!failures.length" @click="exportFailures(false)">
            <Download :size="16" /> 导出全部 MD 包
          </button>
          <button class="secondary-action" type="button" @click="loadFailures"><RefreshCw :size="16" /> 刷新</button>
        </div>
      </div>
      <div class="table-list">
        <div v-for="row in failures" :key="row.id" class="table-row failed failure-row" @click="openFailureReview(row)">
          <input
            type="checkbox"
            :checked="selectedFailureIds.includes(row.id)"
            @click.stop
            @change="updateFailureSelection(row.id, $event.target.checked)"
          />
          <span>#{{ row.id }}</span>
          <strong :title="originalQueryTitle(row)">{{ shortRecordTitle(row, "案例") }}</strong>
          <span>{{ row.disaster_type }} / {{ displayDisasterLabel(row.disaster) }}</span>
          <span>{{ row.top_score ?? "无结果" }}</span>
          <div class="row-actions" @click.stop>
            <button type="button" @click="openFailureReview(row)">Review</button>
          </div>
          <small>{{ row.username }}</small>
        </div>
      </div>
    </section>

    <section v-if="activeTab === 'system'" class="page-panel">
      <div class="page-header">
        <div><h2>系统与索引状态</h2><p>查看本地模型、设备、索引文件和后台构建任务。</p></div>
        <div class="button-row">
          <button class="secondary-action" type="button" :disabled="!!systemAction" @click="loadSystemStatus">
            <Loader2 v-if="systemAction === 'status'" class="spin" :size="16" />
            <RefreshCw v-else :size="16" />
            刷新
          </button>
          <button v-if="canOperateIndex" class="secondary-action" type="button" :disabled="indexBusy" @click="reloadIndex">
            <Loader2 v-if="systemAction === 'reload'" class="spin" :size="16" />
            <Database v-else :size="16" />
            重载索引
          </button>
          <button v-if="canOperateIndex" class="secondary-action" type="button" :disabled="indexBusy" @click="rebuildIndex">
            <Loader2 v-if="systemAction === 'rebuild'" class="spin" :size="16" />
            <FolderCog v-else :size="16" />
            重建完整 hold 索引
          </button>
        </div>
      </div>
      <div
        v-if="systemFeedback"
        :class="[systemFeedback.type === 'error' ? 'error-banner' : 'info-banner', 'inline-banner']"
      >
        <AlertCircle v-if="systemFeedback.type === 'error'" :size="16" />
        <CheckCircle2 v-else :size="16" />
        <span>{{ systemFeedback.message }}</span>
      </div>
      <div v-if="canDevelop" class="tool-panel system-model-panel">
        <label>
          <span>模型 checkpoint</span>
          <select v-model="selectedModelPath" :disabled="!!systemAction">
            <option v-for="item in modelOptions" :key="item.path" :value="item.path">
              {{ modelOptionLabel(item) }}
            </option>
          </select>
        </label>
        <button class="secondary-action" type="button" :disabled="!selectedModelPath || !!systemAction" @click="switchModel">
          <Loader2 v-if="systemAction === 'model'" class="spin" :size="16" />
          <Database v-else :size="16" />
          切换模型
        </button>
      </div>
      <div class="status-grid">
        <div><strong>{{ systemStatus?.retrieval?.ready ? "ready" : "not ready" }}</strong><span>服务状态</span></div>
        <div><strong>{{ systemStatus?.retrieval?.device || health?.device || "-" }}</strong><span>推理设备</span></div>
        <div><strong>{{ systemStatus?.retrieval?.gallery_patch_count || health?.gallery_patch_count || 0 }}</strong><span>图库 patches</span></div>
        <div><strong>{{ indexJob.status || "idle" }}</strong><span>索引任务</span></div>
      </div>
      <div class="system-detail-grid">
        <article>
          <strong>模型与索引</strong>
          <span>current model：{{ currentModelName }}</span>
          <span>scope：{{ systemStatus?.retrieval?.scope || health?.scope || "-" }}</span>
          <span>checkpoint：{{ systemStatus?.retrieval?.checkpoint || "-" }}</span>
          <span>features：{{ systemStatus?.retrieval?.features_path || "-" }}</span>
          <span>metadata：{{ systemStatus?.retrieval?.metadata_path || "-" }}</span>
          <span v-if="systemStatus?.retrieval?.error" class="danger-text">error：{{ systemStatus.retrieval.error }}</span>
        </article>
        <article>
          <strong>后台索引任务</strong>
          <span>status：{{ indexJob.status || "idle" }}</span>
          <span>pid：{{ indexJob.pid || "-" }}</span>
          <span>return code：{{ indexJob.return_code ?? "-" }}</span>
          <span>log：{{ indexJob.log_path || "-" }}</span>
          <span>command：{{ indexJob.command || "-" }}</span>
        </article>
      </div>
      <div class="artifact-list">
        <article v-for="item in systemStatus?.retrieval?.artifacts || []" :key="item.name">
          <strong>{{ item.name }}</strong>
          <span>{{ item.exists ? "存在" : "缺失" }} / {{ item.size_bytes }} bytes</span>
          <small>{{ item.path }}</small>
        </article>
      </div>
    </section>

    <section v-if="activeTab === 'algorithm'" class="page-panel">
      <div class="page-header">
        <div><h2>算法分析</h2><p>面向算法开发人员，汇总模型效果、检索策略和失败样本。</p></div>
        <div class="button-row">
          <button class="secondary-action" type="button" @click="loadAlgorithmAnalysis"><RefreshCw :size="16" /> 刷新</button>
          <button class="secondary-action" type="button" :disabled="!selectedFailureIds.length" @click="exportFailures(true)">
            <FileDown :size="16" /> 导出选中失败案例 MD 包
          </button>
        </div>
      </div>
      <div class="tool-panel">
        <label>
          <span>模型 checkpoint</span>
          <select v-model="selectedModelPath">
            <option v-for="item in modelOptions" :key="item.path" :value="item.path">
              {{ modelOptionLabel(item) }}
            </option>
          </select>
        </label>
        <button class="secondary-action" type="button" :disabled="!selectedModelPath || !!systemAction" @click="switchModel">
          <Loader2 v-if="systemAction === 'model'" class="spin" :size="16" />
          <Database v-else :size="16" />
          切换模型
        </button>
      </div>
      <div class="status-grid">
        <div><strong>{{ algorithmAnalysis?.summary?.total_searches || 0 }}</strong><span>检索次数</span></div>
        <div><strong>{{ algorithmAnalysis?.summary?.avg_top_score ?? "-" }}</strong><span>平均 Top 分数</span></div>
        <div><strong>{{ algorithmAnalysis?.summary?.low_score_count || 0 }}</strong><span>低分记录</span></div>
        <div><strong>{{ algorithmAnalysis?.feedback_counts?.wrong || 0 }}</strong><span>错误反馈</span></div>
      </div>
      <div class="table-list">
        <div v-for="item in algorithmAnalysis?.by_strategy || []" :key="`${item.filter_mode}-${item.aggregation}`" class="table-row static">
          <span>{{ item.filter_mode }}</span>
          <strong>{{ item.aggregation }}</strong>
          <span>{{ item.searches }} searches</span>
          <span>{{ item.avg_top_score ?? "-" }}</span>
          <small>{{ item.avg_elapsed_ms ?? "-" }} ms</small>
        </div>
      </div>
      <div class="table-list">
        <button v-for="row in algorithmAnalysis?.recent_failures || []" :key="row.id" type="button" class="table-row failed algorithm-row" @click="openFailureReview(row)">
          <span>#{{ row.id }}</span>
          <strong :title="originalQueryTitle(row)">{{ shortRecordTitle(row, "案例") }}</strong>
          <span>{{ row.disaster_type }} / {{ displayDisasterLabel(row.disaster) }}</span>
          <span>{{ row.top_score ?? "无结果" }}</span>
          <small>{{ row.username }}</small>
        </button>
      </div>
    </section>

    <section v-if="activeTab === 'users'" class="page-panel">
      <div class="page-header">
        <div><h2>用户管理</h2><p>admin 只能查询和维护 retriever；developer 可查看全部用户信息，但不参与用户编辑。</p></div>
        <div class="button-row">
          <label class="search-inline">
            <Search :size="16" />
            <input v-model="userSearch" placeholder="按用户名查找" @keyup.enter="searchUsers" />
          </label>
          <button class="secondary-action" type="button" @click="searchUsers"><Search :size="16" /> 查找</button>
          <button class="secondary-action" type="button" @click="refreshUsers"><RefreshCw :size="16" /> 刷新</button>
        </div>
      </div>
      <div v-if="userSearchStatus" class="info-banner inline-banner">
        <CheckCircle2 :size="16" />
        <span>{{ userSearchStatus }}</span>
      </div>
      <div v-if="canManageRetrievers" class="tool-panel user-create">
        <label>
          <span>用户名</span>
          <input v-model="userForm.username" />
        </label>
        <label>
          <span>密码</span>
          <input v-model="userForm.password" type="password" />
        </label>
        <label>
          <span>类别</span>
          <select v-model="userForm.role" disabled>
            <option value="retriever">retriever</option>
          </select>
        </label>
        <button class="secondary-action" type="button" @click="createRetrieverUser">
          <Plus :size="16" /> 新增检索用户
        </button>
      </div>
      <div class="table-list">
        <div v-for="item in usersList" :key="item.id" class="table-row static user-row">
          <span>#{{ item.id }}</span>
          <input v-model="item.username" :disabled="!canEditUser(item)" />
          <span>{{ item.role }}</span>
          <input v-model="item.new_password" :disabled="!canEditUser(item)" type="password" placeholder="新密码" />
          <div class="row-actions">
            <button :disabled="!canEditUser(item) || savingUserId === item.id" type="button" @click="saveUser(item)">
              <Save :size="14" /> 保存
            </button>
            <button :disabled="!canEditUser(item) || savingUserId === item.id" type="button" @click="deleteUser(item)">
              <Trash2 :size="14" /> 删除
            </button>
          </div>
        </div>
      </div>
    </section>

    <div v-if="failureReview" class="modal-backdrop" @click.self="closeFailureReview">
      <section class="modal-panel">
        <div class="page-header">
          <div>
            <h2>失败案例复核</h2>
            <p>{{ failureReview.title }} / {{ failureReview.row?.disaster_type }} / {{ displayDisasterLabel(failureReview.row?.disaster) }}</p>
          </div>
          <button class="icon-button" type="button" title="关闭" @click="closeFailureReview">
            <X :size="16" />
          </button>
        </div>

        <div class="compare-grid">
          <figure>
            <img :src="failureReview.queryUrl" alt="" />
            <figcaption>灾后查询图像</figcaption>
          </figure>
          <figure v-if="failureReview.preUrl">
            <img :src="failureReview.preUrl" alt="" />
            <figcaption>灾前候选图像</figcaption>
          </figure>
          <figure v-else class="empty-figure">
            <div class="empty-thumb">无命中灾前图像</div>
            <figcaption>灾前候选图像</figcaption>
          </figure>
        </div>

        <div class="meta-list">
          <span>记录编号：#{{ failureReview.row?.id }}</span>
          <span>失败类型：{{ failureReview.row?.failure_reason || "-" }}</span>
          <span>结果编号：{{ failureReview.item?.search_result_id || "-" }}</span>
          <span>相似度分数：{{ failureReview.item?.score ?? failureReview.row?.top_score ?? "-" }}</span>
        </div>

        <textarea v-model="failureReviewNote" placeholder="Review note"></textarea>
        <div class="review-controls">
          <label>
            <span>Feedback</span>
            <select v-model="failureReviewFeedback" :disabled="!failureReview.item?.search_result_id || savingFailureReview">
              <option value="correct">correct</option>
              <option value="wrong">wrong</option>
              <option value="uncertain">uncertain</option>
            </select>
          </label>
          <button
            class="secondary-action"
            type="button"
            :disabled="!failureReview.item?.search_result_id || savingFailureReview"
            @click="submitFailureFeedback"
          >
            <Save :size="15" /> {{ savingFailureReview ? "Saving..." : "Confirm Update" }}
          </button>
        </div>
      </section>
    </div>
  </main>
</template>
