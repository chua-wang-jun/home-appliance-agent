import json
import threading
from pathlib import Path


class SourceStorage:
    """保存每个会话线程命中过的知识来源文件。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def append(self, thread_id: str, sources: list[str]) -> None:
        with self.lock:
            data = self._read()
            entries = data.get(thread_id, [])
            if not isinstance(entries, list):
                entries = []
            entries.append(list(dict.fromkeys(sources)))
            data[thread_id] = entries
            self._write(data)

    def get(self, thread_id: str) -> list[list[str]]:
        with self.lock:
            entries = self._read().get(thread_id, [])
            return entries if isinstance(entries, list) else []

    def delete(self, thread_id: str) -> None:
        with self.lock:
            data = self._read()
            if thread_id not in data:
                return
            data.pop(thread_id, None)
            self._write(data)
