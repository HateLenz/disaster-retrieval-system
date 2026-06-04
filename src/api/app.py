from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.app_store import LocalAppStore, ROLES
from src.api.retrieval_service import DisasterRetrievalService, public_asset_url


class AuthRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = Field(default="retriever")


class SearchByPathRequest(BaseModel):
    image_path: str = Field(..., description="Repository-relative or known xBD image path.")
    disaster_type: str | None = None
    disaster: str | None = None
    top_k: int = 5
    filter_mode: str = "both"
    aggregation: str = "top_m"


class BatchPathSearchRequest(BaseModel):
    image_paths: list[str]
    disaster_type: str | None = None
    disaster: str | None = None
    top_k: int = 5
    filter_mode: str = "both"
    aggregation: str = "top_m"


class FeedbackRequest(BaseModel):
    search_result_id: int
    feedback: str
    note: str | None = None


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "retriever"


class UserUpdateRequest(BaseModel):
    username: str | None = None
    role: str | None = None
    password: str | None = None


class RebuildIndexRequest(BaseModel):
    split: str = "hold"
    amp: bool = True


class SwitchModelRequest(BaseModel):
    checkpoint_path: str


app = FastAPI(
    title="Disaster Pre-Disaster Image Retrieval API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "app://.",
        "file://",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

service = DisasterRetrievalService()
store = LocalAppStore()
index_job: dict[str, Any] = {
    "process": None,
    "status": "idle",
    "pid": None,
    "command": None,
}


@app.on_event("startup")
def load_application() -> None:
    store.initialize()
    service.load()


def _clean_filter(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _service_error(error: Exception) -> HTTPException:
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, PermissionError):
        return HTTPException(status_code=403, detail=str(error))
    if isinstance(error, (RuntimeError, ValueError)):
        status_code = 503 if not service.ready else 400
        return HTTPException(status_code=status_code, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


def _parse_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Bearer token is required.")
    return token.strip()


def current_user(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    token = _parse_bearer(authorization)
    user = store.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    return user


def current_user_for_download(
    authorization: Annotated[str | None, Header()] = None,
    token: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    if token is not None and token.strip():
        user = store.get_user_by_token(token.strip())
        if user is None:
            raise HTTPException(status_code=401, detail="Session is invalid or expired.")
        return user
    return current_user(authorization)


def require_roles(*roles: str):
    def dependency(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict[str, Any]:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role permission.")
        return user

    return dependency


def _auth_payload(user: dict[str, Any]) -> dict[str, Any]:
    token = store.create_session(user["id"])
    return {"token": token, "user": user, "roles": list(ROLES)}


def _save_upload(filename: str, image_bytes: bytes) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename or "query.png").strip("._") or "query.png"
    output_dir = REPO_ROOT / "outputs" / "uploads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_name
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{output_path.stem}_{counter}{output_path.suffix}"
        counter += 1
    output_path.write_bytes(image_bytes)
    return str(output_path.relative_to(REPO_ROOT)).replace("\\", "/")


def _persist_search(
    user: dict[str, Any],
    payload: dict[str, Any],
    query_kind: str,
    query_label: str,
    disaster_type: str | None,
    disaster: str | None,
    filter_mode: str,
    aggregation: str,
    top_k: int,
) -> dict[str, Any]:
    return store.create_search_record(
        user_id=int(user["id"]),
        query_kind=query_kind,
        query_label=query_label,
        disaster_type=disaster_type,
        disaster=disaster,
        filter_mode=filter_mode,
        aggregation=aggregation,
        top_k=top_k,
        payload=payload,
    )


def _index_job_status() -> dict[str, Any]:
    process = index_job.get("process")
    if process is not None:
        return_code = process.poll()
        if return_code is None:
            index_job["status"] = "running"
        else:
            index_job["status"] = "succeeded" if return_code == 0 else "failed"
            index_job["return_code"] = return_code
            index_job["process"] = None
    return {key: value for key, value in index_job.items() if key != "process"}


def _system_status_payload() -> dict[str, Any]:
    job = _index_job_status()
    return {"retrieval": service.diagnostics(), "index_job": job, "job": job}


def _parse_ids(raw_ids: str | None) -> list[int] | None:
    if raw_ids is None or raw_ids.strip() == "":
        return None
    ids: list[int] = []
    for value in raw_ids.split(","):
        value = value.strip()
        if not value:
            continue
        ids.append(int(value))
    return ids or None


def _slugify(value: str | None, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._-")
    return clean or fallback


def _display_query_label(query_label: str | None, disaster: str | None = None) -> str:
    base = Path(query_label or disaster or "query").stem
    lower = base.lower()
    markers = ("fire", "flood", "flooding", "wind", "earthquake", "hurricane", "tsunami", "volcano")
    cut_at = len(base)
    for marker in markers:
        marker_index = lower.find(marker)
        if marker_index > 0:
            cut_at = min(cut_at, marker_index)
    display = re.sub(r"[_-]+", " ", base[:cut_at]).strip()
    return display or (disaster or base)


def _failure_reason_text(reason: str | None) -> str:
    mapping = {
        "low_score": "低分检索",
        "no_result": "无结果",
        "wrong_feedback": "人工判定错误",
    }
    return mapping.get(reason or "", reason or "-")


def _zip_asset(archive: zipfile.ZipFile, folder: str, raw_path: str | None, name: str) -> str | None:
    if not raw_path:
        return None
    try:
        asset_path = service.resolve_asset_path(raw_path)
    except (FileNotFoundError, PermissionError):
        return None
    extension = asset_path.suffix or ".png"
    relative_path = f"images/{name}{extension.lower()}"
    archive.write(asset_path, f"{folder}/{relative_path}")
    return relative_path


def _markdown_asset_url(request: Request, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    try:
        service.resolve_asset_path(raw_path)
    except (FileNotFoundError, PermissionError):
        return None
    return f"{str(request.base_url).rstrip('/')}{public_asset_url(raw_path)}"


def _history_report_markdown(
    detail: dict[str, Any],
    query_image: str | None,
    result_images: list[tuple[dict[str, Any], str | None]],
    failure: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# 灾后检索报告",
        "",
        f"- 检索编号：{detail['id']}",
        f"- 检索时间：{detail['query_time']}",
        f"- 用户名：{detail['username']}",
        f"- 图像名称：{_display_query_label(detail.get('query_label'), detail.get('disaster'))}",
        f"- 灾害类型：{detail.get('disaster_type') or '-'}",
        f"- 灾害地区：{detail.get('disaster') or '-'}",
        f"- 过滤模式：{detail['filter_mode']}",
        f"- 聚合策略：{detail['aggregation']}",
        f"- 返回数量：{detail['result_count']}",
        f"- 检索耗时：{detail['elapsed_ms']} ms",
        f"- 索引范围：{detail.get('scope') or '-'}",
    ]
    if failure is not None:
        lines.extend(
            [
                "",
                "## 失败案例信息",
                "",
                f"- 失败类型：{_failure_reason_text(failure.get('failure_reason'))}",
                f"- 反馈标签：{failure.get('feedback') or '-'}",
                f"- 反馈备注：{failure.get('feedback_note') or '-'}",
            ]
        )
    if query_image:
        lines.extend(["", "## 灾后查询图像", "", f"![query]({query_image})"])

    lines.extend(["", "## Top 结果", ""])
    if not result_images:
        lines.append("未命中灾前图像。")
        return "\n".join(lines)

    for item, image_path in result_images:
        lines.extend(
            [
                f"### Top {item.get('rank')}",
                "",
                f"- 相似度分数：{item.get('score')}",
                f"- 最优 patch 分数：{item.get('best_patch_score')}",
                f"- 灾前图像编号：{item.get('tile_id') or '-'}",
                f"- 建筑编号：{item.get('building_uid') or '-'}",
                f"- 灾害地区：{item.get('disaster') or '-'}",
                f"- 灾害类型：{item.get('disaster_type') or '-'}",
                f"- 损毁等级：{item.get('damage_label') or '-'}",
            ]
        )
        if image_path:
            lines.extend(["", f"![top_{item.get('rank')}]({image_path})"])
        lines.append("")
    return "\n".join(lines)


def _build_history_report_markdown(user: dict[str, Any], search_id: int, request: Request) -> tuple[str, str]:
    detail = store.get_history_detail(search_id, user)
    query = detail.get("query", {}) or {}
    query_image = _markdown_asset_url(request, query.get("source_path"))
    result_images: list[tuple[dict[str, Any], str | None]] = []
    for item in detail.get("results", []):
        image_url = _markdown_asset_url(request, item.get("pre_patch_path") or item.get("pre_image_path"))
        result_images.append((item, image_url))

    report = _history_report_markdown(detail, query_image=query_image, result_images=result_images)
    label = _slugify(_display_query_label(detail.get("query_label"), detail.get("disaster")), f"search_{search_id}")
    return report, f"retrieval_report_{search_id}_{label}.md"


def _build_export_zip(
    user: dict[str, Any],
    search_ids: list[int],
    *,
    failure_rows: dict[int, dict[str, Any]] | None = None,
    filename_prefix: str,
) -> Path:
    export_dir = REPO_ROOT / "outputs" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    zip_path = export_dir / f"{filename_prefix}_{timestamp}.zip"
    failure_rows = failure_rows or {}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if not search_ids:
            archive.writestr("README.md", "# 空导出\n\n当前没有可导出的记录。\n")
            return zip_path

        for search_id in search_ids:
            detail = store.get_history_detail(search_id, user)
            folder = f"search_{search_id:04d}_{_slugify(_display_query_label(detail.get('query_label'), detail.get('disaster')), f'search_{search_id}')}"
            query = detail.get("query", {}) or {}
            query_image = _zip_asset(archive, folder, query.get("source_path"), "query")

            result_images: list[tuple[dict[str, Any], str | None]] = []
            for item in detail.get("results", []):
                result_image = _zip_asset(
                    archive,
                    folder,
                    item.get("pre_patch_path") or item.get("pre_image_path"),
                    f"result_{int(item.get('rank', 0)):02d}_pre",
                )
                result_images.append((item, result_image))

            report = _history_report_markdown(
                detail,
                query_image=query_image,
                result_images=result_images,
                failure=failure_rows.get(search_id),
            )
            archive.writestr(f"{folder}/report.md", report)

    return zip_path


def _assert_admin_can_manage_retriever(target: dict[str, Any]) -> None:
    if target["role"] != "retriever":
        raise HTTPException(status_code=403, detail="Administrators can only manage retriever accounts.")


@app.get("/api/health")
def health() -> dict:
    return service.status()


@app.post("/api/auth/register")
def register(payload: RegisterRequest) -> dict:
    try:
        user = store.create_user(
            username=payload.username,
            password=payload.password,
            role=payload.role,
            display_name=payload.username,
        )
        return _auth_payload(user)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/auth/login")
def login(payload: AuthRequest) -> dict:
    try:
        user = store.authenticate(payload.username, payload.password)
        return _auth_payload(user)
    except ValueError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


@app.get("/api/auth/me")
def me(user: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    return {"user": user, "roles": list(ROLES)}


@app.post("/api/auth/logout")
def logout(authorization: Annotated[str | None, Header()] = None) -> dict:
    token = _parse_bearer(authorization)
    store.revoke_session(token)
    return {"ok": True}


@app.get("/api/users")
def users(
    user: Annotated[dict[str, Any], Depends(require_roles("admin", "developer"))],
    query: Annotated[str | None, Query()] = None,
) -> dict:
    return {"users": store.list_users(user, query=query), "roles": list(ROLES)}


@app.post("/api/users")
def create_user(
    payload: UserCreateRequest,
    _: Annotated[dict[str, Any], Depends(require_roles("admin"))],
) -> dict:
    if payload.role != "retriever":
        raise HTTPException(status_code=403, detail="Administrators can only create retriever accounts.")
    try:
        return {
            "user": store.create_user(
                username=payload.username,
                password=payload.password,
                role="retriever",
                display_name=payload.username,
            )
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.patch("/api/users/{user_id}")
def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    _: Annotated[dict[str, Any], Depends(require_roles("admin"))],
) -> dict:
    try:
        target = store.get_user_by_id(user_id)
        _assert_admin_can_manage_retriever(target)
        if payload.role is not None and payload.role != "retriever":
            raise HTTPException(status_code=403, detail="Administrators can only manage retriever accounts.")
        return {
            "user": store.update_user(
                user_id,
                username=payload.username,
                role=payload.role,
                password=payload.password,
            )
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/users/{user_id}")
def delete_user(
    user_id: int,
    _: Annotated[dict[str, Any], Depends(require_roles("admin"))],
) -> dict:
    try:
        target = store.get_user_by_id(user_id)
        _assert_admin_can_manage_retriever(target)
        return {"user": store.deactivate_user(user_id), "deleted": True}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/options")
def options(_: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    try:
        return service.options()
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/samples")
def samples(
    _: Annotated[dict[str, Any], Depends(current_user)],
    disaster_type: Annotated[str | None, Query()] = None,
    disaster: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 12,
) -> dict:
    try:
        return {
            "samples": service.samples(
                disaster_type=_clean_filter(disaster_type),
                disaster=_clean_filter(disaster),
                limit=limit,
            )
        }
    except Exception as error:
        raise _service_error(error) from error


@app.post("/api/search")
async def search(
    user: Annotated[dict[str, Any], Depends(current_user)],
    image: Annotated[UploadFile, File()],
    disaster_type: Annotated[str | None, Form()] = None,
    disaster: Annotated[str | None, Form()] = None,
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
    filter_mode: Annotated[str, Form()] = "both",
    aggregation: Annotated[str, Form()] = "top_m",
) -> dict:
    try:
        image_bytes = await image.read()
        saved_path = _save_upload(image.filename or "query.png", image_bytes)
        payload = service.search_bytes(
            image_bytes=image_bytes,
            disaster_type=_clean_filter(disaster_type),
            disaster=_clean_filter(disaster),
            top_k=top_k,
            filter_mode=filter_mode,
            aggregation=aggregation,
        )
        payload["query"]["source_path"] = saved_path
        payload["query"]["source_url"] = public_asset_url(saved_path)
        return _persist_search(
            user,
            payload,
            query_kind="upload",
            query_label=image.filename or saved_path,
            disaster_type=_clean_filter(disaster_type),
            disaster=_clean_filter(disaster),
            filter_mode=filter_mode,
            aggregation=aggregation,
            top_k=top_k,
        )
    except Exception as error:
        raise _service_error(error) from error


@app.post("/api/search-by-path")
def search_by_path(payload: SearchByPathRequest, user: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    try:
        result = service.search_path(
            image_path=payload.image_path,
            disaster_type=_clean_filter(payload.disaster_type),
            disaster=_clean_filter(payload.disaster),
            top_k=payload.top_k,
            filter_mode=payload.filter_mode,
            aggregation=payload.aggregation,
        )
        return _persist_search(
            user,
            result,
            query_kind="path",
            query_label=payload.image_path,
            disaster_type=_clean_filter(payload.disaster_type),
            disaster=_clean_filter(payload.disaster),
            filter_mode=payload.filter_mode,
            aggregation=payload.aggregation,
            top_k=payload.top_k,
        )
    except Exception as error:
        raise _service_error(error) from error


@app.post("/api/batch-search")
async def batch_search(
    user: Annotated[dict[str, Any], Depends(current_user)],
    images: Annotated[list[UploadFile], File()],
    disaster_type: Annotated[str | None, Form()] = None,
    disaster: Annotated[str | None, Form()] = None,
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
    filter_mode: Annotated[str, Form()] = "both",
    aggregation: Annotated[str, Form()] = "top_m",
) -> dict:
    batch_results = []
    for image in images[:20]:
        try:
            image_bytes = await image.read()
            saved_path = _save_upload(image.filename or "query.png", image_bytes)
            payload = service.search_bytes(
                image_bytes=image_bytes,
                disaster_type=_clean_filter(disaster_type),
                disaster=_clean_filter(disaster),
                top_k=top_k,
                filter_mode=filter_mode,
                aggregation=aggregation,
            )
            payload["query"]["source_path"] = saved_path
            payload["query"]["source_url"] = public_asset_url(saved_path)
            payload = _persist_search(
                user,
                payload,
                query_kind="batch_upload",
                query_label=image.filename or saved_path,
                disaster_type=_clean_filter(disaster_type),
                disaster=_clean_filter(disaster),
                filter_mode=filter_mode,
                aggregation=aggregation,
                top_k=top_k,
            )
            batch_results.append({"filename": image.filename, "ok": True, "result": payload})
        except Exception as error:
            batch_results.append({"filename": image.filename, "ok": False, "error": str(error)})
    return {"count": len(batch_results), "items": batch_results}


@app.post("/api/batch-search-by-path")
def batch_search_by_path(payload: BatchPathSearchRequest, user: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    batch_results = []
    for image_path in payload.image_paths[:50]:
        try:
            result = service.search_path(
                image_path=image_path,
                disaster_type=_clean_filter(payload.disaster_type),
                disaster=_clean_filter(payload.disaster),
                top_k=payload.top_k,
                filter_mode=payload.filter_mode,
                aggregation=payload.aggregation,
            )
            result = _persist_search(
                user,
                result,
                query_kind="batch_path",
                query_label=image_path,
                disaster_type=_clean_filter(payload.disaster_type),
                disaster=_clean_filter(payload.disaster),
                filter_mode=payload.filter_mode,
                aggregation=payload.aggregation,
                top_k=payload.top_k,
            )
            batch_results.append({"image_path": image_path, "ok": True, "result": result})
        except Exception as error:
            batch_results.append({"image_path": image_path, "ok": False, "error": str(error)})
    return {"count": len(batch_results), "items": batch_results}


@app.get("/api/history")
def history(
    user: Annotated[dict[str, Any], Depends(current_user)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict:
    return {"history": store.list_history(user, limit=limit)}


@app.get("/api/history/export")
def history_export(
    user: Annotated[dict[str, Any], Depends(current_user_for_download)],
    ids: Annotated[str | None, Query()] = None,
) -> FileResponse:
    try:
        selected_ids = _parse_ids(ids) or [row["id"] for row in store.list_history(user, limit=1000)]
        zip_path = _build_export_zip(user, selected_ids[:500], filename_prefix="retrieval_history")
        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/history/{search_id}")
def history_detail(search_id: int, user: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    try:
        return store.get_history_detail(search_id, user)
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/history/{search_id}/report")
def history_report(search_id: int, user: Annotated[dict[str, Any], Depends(current_user_for_download)]) -> FileResponse:
    try:
        zip_path = _build_export_zip(user, [search_id], filename_prefix=f"retrieval_report_{search_id}")
        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/history/{search_id}/report.md")
def history_report_markdown(
    search_id: int,
    request: Request,
    user: Annotated[dict[str, Any], Depends(current_user_for_download)],
) -> Response:
    try:
        report, filename = _build_history_report_markdown(user, search_id, request)
        return Response(
            content=report,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/results/{result_id}")
def result_detail(result_id: int, user: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    try:
        return store.get_result_detail(result_id, user)
    except Exception as error:
        raise _service_error(error) from error


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest, user: Annotated[dict[str, Any], Depends(current_user)]) -> dict:
    try:
        store.get_result_detail(payload.search_result_id, user)
        return store.add_feedback(
            search_result_id=payload.search_result_id,
            user_id=int(user["id"]),
            feedback=payload.feedback,
            note=payload.note,
            replace_result_feedback=user["role"] in {"admin", "developer"},
        )
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/failures")
def failures(
    user: Annotated[dict[str, Any], Depends(require_roles("admin", "developer"))],
    score_threshold: Annotated[float, Query(ge=0, le=1)] = 0.25,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> dict:
    return {"failures": store.list_failures(user, score_threshold=score_threshold, limit=limit)}


@app.get("/api/failures/export")
def failures_export(
    user: Annotated[dict[str, Any], Depends(current_user_for_download)],
    ids: Annotated[str | None, Query()] = None,
    score_threshold: Annotated[float, Query(ge=0, le=1)] = 0.35,
) -> FileResponse:
    try:
        if user["role"] not in {"admin", "developer"}:
            raise HTTPException(status_code=403, detail="Insufficient role permission.")
        failure_rows = store.list_failures(user, score_threshold=score_threshold, limit=1000)
        selected_ids = _parse_ids(ids)
        if selected_ids:
            selected = set(selected_ids)
            failure_rows = [row for row in failure_rows if int(row["id"]) in selected]
        zip_path = _build_export_zip(
            user,
            [int(row["id"]) for row in failure_rows[:500]],
            failure_rows={int(row["id"]): row for row in failure_rows[:500]},
            filename_prefix="retrieval_failures",
        )
        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)
    except Exception as error:
        raise _service_error(error) from error


@app.get("/api/system/status")
def system_status(_: Annotated[dict[str, Any], Depends(require_roles("admin", "developer"))]) -> dict:
    return _system_status_payload()


@app.get("/api/index/status")
def index_status(_: Annotated[dict[str, Any], Depends(require_roles("admin", "developer"))]) -> dict:
    return _system_status_payload()


@app.get("/api/algorithm/analysis")
def algorithm_analysis(
    _: Annotated[dict[str, Any], Depends(require_roles("developer"))],
    score_threshold: Annotated[float, Query(ge=0, le=1)] = 0.35,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict:
    return store.algorithm_analysis(score_threshold=score_threshold, limit=limit)


@app.get("/api/models")
def models(_: Annotated[dict[str, Any], Depends(require_roles("developer"))]) -> dict:
    return {"models": service.available_checkpoints(), "current": service.status().get("checkpoint")}


@app.post("/api/models/switch")
def switch_model(
    payload: SwitchModelRequest,
    _: Annotated[dict[str, Any], Depends(require_roles("developer"))],
) -> dict:
    try:
        service.switch_checkpoint(payload.checkpoint_path)
        return _system_status_payload()
    except Exception as error:
        raise _service_error(error) from error


@app.post("/api/index/reload")
def index_reload(_: Annotated[dict[str, Any], Depends(require_roles("admin", "developer"))]) -> dict:
    service.load()
    return _system_status_payload()


@app.post("/api/index/rebuild")
def index_rebuild(
    payload: RebuildIndexRequest,
    _: Annotated[dict[str, Any], Depends(require_roles("admin", "developer"))],
) -> dict:
    status = _index_job_status()
    if status.get("status") == "running":
        raise HTTPException(status_code=409, detail="Index build is already running.")

    command = [sys.executable, "scripts/build_api_index.py", "--split", payload.split]
    if payload.amp:
        command.append("--amp")
    log_path = REPO_ROOT / "outputs" / "logs" / "api_index_rebuild.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    index_job.update(
        {
            "process": process,
            "status": "running",
            "pid": process.pid,
            "command": " ".join(command),
            "log_path": str(log_path),
            "return_code": None,
        }
    )
    return _system_status_payload()


@app.get("/api/assets")
def asset(path: Annotated[str, Query()]) -> FileResponse:
    try:
        return FileResponse(service.resolve_asset_path(path))
    except Exception as error:
        raise _service_error(error) from error


frontend_dist = REPO_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
