import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import openai
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import rag as rag_service
from utils.file_handler import docx_loader
from utils.logger_handler import logger

HTML_PATH = PROJECT_ROOT / "web" / "index.html"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "agent_memory"
SOURCES_PATH = CHECKPOINT_DIR / "sources.json"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
KB_LOCK = threading.Lock()
SOURCES_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
agent: ReactAgent | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None


class KnowledgeRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=500000)


def new_conversation() -> dict:
    return {"id": uuid4().hex, "title": "新对话", "messages": []}


def conversation_title(messages: list[dict]) -> str:
    first_user = next(
        (item["content"] for item in messages if item["role"] == "user"),
        "新对话",
    )
    return " ".join(first_user.split())[:36] or "新对话"


def list_knowledge_files() -> list[dict]:
    if not KB_DIR.exists():
        return []
    chunk_counts = rag_service.vector_store.get_source_chunk_counts()
    files = []
    for path in sorted(KB_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "editable": path.suffix.lower() == ".txt",
                "type": path.suffix.lower().lstrip("."),
                "chunk_count": chunk_counts.get(str(path.resolve()), 0),
            }
        )
    return files


def safe_knowledge_path(filename: str, *, text_only: bool = False) -> Path:
    name = Path(filename).name
    if name != filename or not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="文件名不合法")
    path = (KB_DIR / name).resolve()
    if path.parent != KB_DIR.resolve():
        raise HTTPException(status_code=400, detail="文件名不合法")
    suffix = path.suffix.lower()
    allowed = {".txt"} if text_only else {".txt", ".pdf", ".docx"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="仅支持 TXT、PDF 和 DOCX 文件")
    return path


def normalize_text_filename(filename: str) -> str:
    name = filename.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if Path(name).suffix == "":
        name += ".txt"
    if Path(name).suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="网页端只支持新增和编辑 TXT 文件")
    safe_knowledge_path(name, text_only=True)
    return name


def rebuild_knowledge_index() -> int:
    loaded_count = rag_service.vector_store.rebuild()
    rag_service.retriever = rag_service.vector_store.get_retriever()
    return loaded_count


def media_type_for(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def load_sources_map() -> dict:
    if not SOURCES_PATH.exists():
        return {}
    try:
        data = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        logger.warning("[sources]读取来源记录失败", exc_info=True)
        return {}


def save_thread_sources(thread_id: str, sources: list[str]) -> None:
    with SOURCES_LOCK:
        data = load_sources_map()
        entries = data.get(thread_id, [])
        if not isinstance(entries, list):
            entries = []
        entries.append(list(dict.fromkeys(sources)))
        data[thread_id] = entries
        temp_path = SOURCES_PATH.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(SOURCES_PATH)


def get_thread_sources(thread_id: str) -> list[list[str]]:
    data = load_sources_map()
    entries = data.get(thread_id, [])
    return entries if isinstance(entries, list) else []


def delete_thread_sources(thread_id: str) -> None:
    with SOURCES_LOCK:
        data = load_sources_map()
        if thread_id in data:
            data.pop(thread_id, None)
            temp_path = SOURCES_PATH.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(SOURCES_PATH)


def sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global agent
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    agent = ReactAgent()
    yield
    if agent is not None:
        agent.close()
    agent = None


app = FastAPI(title="家庭设备管家", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "agent_ready": agent is not None,
        "memory": str(CHECKPOINT_DIR / "checkpoints.db"),
    }


@app.get("/api/conversations")
def api_list_conversations() -> list[dict]:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    return agent.list_threads()


@app.post("/api/conversations")
def api_create_conversation() -> dict:
    return new_conversation()


@app.get("/api/conversations/{conversation_id}")
def api_get_conversation(conversation_id: str) -> dict:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    messages = agent.get_thread_messages(conversation_id)
    return {
        "id": conversation_id,
        "title": conversation_title(messages),
        "messages": messages,
        "sources": get_thread_sources(conversation_id),
    }


@app.delete("/api/conversations/{conversation_id}")
def api_delete_conversation(conversation_id: str) -> dict:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    agent.delete_thread(conversation_id)
    delete_thread_sources(conversation_id)
    return {"ok": True}


@app.get("/api/knowledge")
def api_list_knowledge() -> dict:
    return {
        "directory": str(KB_DIR),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "files": list_knowledge_files(),
    }


@app.get("/api/knowledge/file")
def api_get_knowledge_file(filename: str) -> dict:
    path = safe_knowledge_path(filename, text_only=True)
    if not path.exists():
        raise HTTPException(status_code=404, detail="知识文件不存在")
    return {
        "filename": path.name,
        "content": path.read_text(encoding="utf-8"),
        "files": list_knowledge_files(),
    }


@app.post("/api/knowledge/file")
def api_save_knowledge_file(payload: KnowledgeRequest) -> dict:
    filename = normalize_text_filename(payload.filename)
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="知识内容不能为空")

    with KB_LOCK:
        path = safe_knowledge_path(filename, text_only=True)
        path.write_text(content + "\n", encoding="utf-8")
        loaded_count = rebuild_knowledge_index()

    return {
        "ok": True,
        "filename": filename,
        "loaded_count": loaded_count,
        "files": list_knowledge_files(),
    }


@app.post("/api/knowledge/upload")
async def api_upload_knowledge_file(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="上传仅支持 PDF 和 DOCX 文件")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件不能超过 30MB")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="文件不是有效的 PDF")
    if suffix == ".docx" and not content.startswith(b"PK"):
        raise HTTPException(status_code=400, detail="文件不是有效的 DOCX")

    path = safe_knowledge_path(filename)
    with KB_LOCK:
        path.write_bytes(content)
        loaded_count = rebuild_knowledge_index()

    return {
        "ok": True,
        "filename": filename,
        "loaded_count": loaded_count,
        "files": list_knowledge_files(),
    }


@app.delete("/api/knowledge/file")
def api_delete_knowledge_file(filename: str) -> dict:
    path = safe_knowledge_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="知识文件不存在")

    with KB_LOCK:
        path.unlink()
        loaded_count = rebuild_knowledge_index()

    return {
        "ok": True,
        "filename": filename,
        "loaded_count": loaded_count,
        "files": list_knowledge_files(),
    }


@app.get("/api/knowledge/preview")
def api_preview_knowledge_file(filename: str) -> dict:
    path = safe_knowledge_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="知识文件不存在")

    suffix = path.suffix.lower()
    chunk_counts = rag_service.vector_store.get_source_chunk_counts()
    result = {
        "filename": path.name,
        "type": suffix.lstrip("."),
        "size": path.stat().st_size,
        "chunk_count": chunk_counts.get(str(path.resolve()), 0),
        "download_url": f"/api/knowledge/download?filename={quote(path.name)}",
    }
    if suffix == ".pdf":
        result["preview_url"] = f"/api/knowledge/raw?filename={quote(path.name)}"
        return result
    if suffix == ".docx":
        documents = docx_loader(str(path))
        result["content"] = "\n".join(doc.page_content for doc in documents)
        return result
    result["content"] = path.read_text(encoding="utf-8")
    return result


@app.get("/api/knowledge/raw")
def api_raw_knowledge_file(filename: str):
    path = safe_knowledge_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="知识文件不存在")
    return FileResponse(
        path,
        media_type=media_type_for(path),
        content_disposition_type="inline",
    )


@app.get("/api/knowledge/download")
def api_download_knowledge_file(filename: str):
    path = safe_knowledge_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="知识文件不存在")
    return FileResponse(
        path,
        filename=path.name,
        media_type=media_type_for(path),
        content_disposition_type="attachment",
    )


@app.post("/api/knowledge/rebuild")
def api_rebuild_knowledge() -> dict:
    with KB_LOCK:
        loaded_count = rebuild_knowledge_index()
    return {
        "ok": True,
        "loaded_count": loaded_count,
        "files": list_knowledge_files(),
    }


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")

    thread_id = payload.conversation_id or uuid4().hex
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="消息不能为空")

    def stream():
        yield sse_event({"type": "meta", "conversation_id": thread_id})
        answer_parts: list[str] = []
        rag_service.reset_sources()
        try:
            for attempt in range(2):
                try:
                    for chunk in agent.execute_chat_stream(
                        question,
                        thread_id=thread_id,
                    ):
                        if not chunk:
                            continue
                        answer_parts.append(chunk)
                        yield sse_event({"type": "delta", "content": chunk})
                    break
                except openai.APIConnectionError:
                    if answer_parts or attempt == 1:
                        raise
                    logger.warning("[web chat]模型连接失败，1 秒后重试")
                    time.sleep(1)

            sources = rag_service.get_sources()
            save_thread_sources(thread_id, sources)
            yield sse_event({
                "type": "done",
                "conversation_id": thread_id,
                "sources": sources,
            })
        except openai.APIConnectionError:
            logger.exception("[web chat]模型服务连接失败")
            yield sse_event({
                "type": "error",
                "message": "模型服务连接失败，请检查网络、DNS 或代理设置后重试。",
            })
        except Exception as exc:
            logger.exception("[web chat]生成回答失败")
            yield sse_event({"type": "error", "message": f"生成失败：{exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
