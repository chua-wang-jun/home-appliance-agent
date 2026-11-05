import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .storage import SourceStorage


class ChatMemory:
    """统一管理 LangGraph Checkpointer、线程消息和来源记录。"""

    def __init__(self, data_dir: str | Path | None = None):
        default_dir = Path(__file__).resolve().parent / "data"
        self.data_dir = Path(data_dir) if data_dir else default_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_db = self.data_dir / "checkpoints.db"
        self.sources_path = self.data_dir / "sources.json"

        self.checkpoint_conn = sqlite3.connect(
            self.checkpoint_db,
            check_same_thread=False,
            timeout=15,
        )
        self.checkpointer = SqliteSaver(self.checkpoint_conn)
        self.sources = SourceStorage(self.sources_path)

    def close(self) -> None:
        self.checkpoint_conn.close()

    @staticmethod
    def thread_config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id or "default"}}

    @staticmethod
    def _message_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def get_thread_messages(
        self,
        agent: Any,
        thread_id: str,
        limit: int = 200,
    ) -> list[dict]:
        state = agent.get_state(self.thread_config(thread_id))
        raw_messages = state.values.get("messages", []) if state.values else []
        messages = []

        for message in raw_messages:
            if isinstance(message, HumanMessage):
                content = self._message_text(message.content).strip()
                if content:
                    messages.append({"role": "user", "content": content})
            elif isinstance(message, AIMessage):
                content = self._message_text(message.content).strip()
                if content:
                    messages.append({"role": "assistant", "content": content})

        return messages[-limit:]

    def list_threads(self, agent: Any, checkpoint_limit: int = 500) -> list[dict]:
        latest_checkpoints = {}
        for checkpoint in self.checkpointer.list(None, limit=checkpoint_limit):
            thread_id = checkpoint.config["configurable"]["thread_id"]
            if thread_id not in latest_checkpoints:
                latest_checkpoints[thread_id] = checkpoint

        conversations = []
        for thread_id, checkpoint in latest_checkpoints.items():
            messages = self.get_thread_messages(agent, thread_id)
            if not messages:
                continue
            first_user = next(
                (item["content"] for item in messages if item["role"] == "user"),
                "新对话",
            )
            conversations.append(
                {
                    "id": thread_id,
                    "title": " ".join(first_user.split())[:36] or "新对话",
                    "updated_at": str(checkpoint.checkpoint.get("ts", "")),
                    "message_count": len(messages),
                    "last_message": messages[-1]["content"][:120],
                }
            )

        conversations.sort(key=lambda item: item["updated_at"], reverse=True)
        return conversations

    def save_thread_sources(self, thread_id: str, sources: list[str]) -> None:
        self.sources.append(thread_id, sources)

    def get_thread_sources(self, thread_id: str) -> list[list[str]]:
        return self.sources.get(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)
        self.sources.delete(thread_id)
