"""
Blob storage abstraction. Live Azure Blob when AZURE_STORAGE_CONNECTION_STRING is set, otherwise a local
filesystem mock (default for local runs/tests; root = .mock_blob/<container> or $MOCK_BLOB_DIR).
Only the five primitives below touch the backend; every typed helper is built on them, so both backends
behave identically (including overwrite=False -> returns False when the blob already exists).
"""
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class BlobStorageClient:
    def __init__(self, connection_string: Optional[str] = None, container: Optional[str] = None,
                 base_dir: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.container = container or os.environ.get("AZURE_STORAGE_CONTAINER", "drivers-pipeline")
        self.is_mock = not bool(self.connection_string)
        root = base_dir or os.environ.get("MOCK_BLOB_DIR")
        self.base_dir = Path(root) if root else Path(__file__).resolve().parents[2] / ".mock_blob" / self.container
        self._live = None  # lazily created azure ContainerClient (the SDK retries transient errors itself)

    # -- backend ----------------------------------------------------------
    def _container(self):
        if self._live is None:
            from azure.storage.blob import ContainerClient
            self._live = ContainerClient.from_connection_string(self.connection_string, self.container)
        return self._live

    def _path(self, blob_path: str) -> Path:
        return self.base_dir / blob_path.replace("\\", "/").lstrip("/")

    @staticmethod
    def _key(blob_path: str) -> str:
        return blob_path.replace("\\", "/").lstrip("/")

    # -- primitives -------------------------------------------------------
    def exists(self, blob_path: str) -> bool:
        if self.is_mock:
            return self._path(blob_path).is_file()
        return self._container().get_blob_client(self._key(blob_path)).exists()

    def read_bytes(self, blob_path: str) -> bytes:
        if self.is_mock:
            return self._path(blob_path).read_bytes()
        return self._container().download_blob(self._key(blob_path)).readall()

    def write_bytes(self, blob_path: str, data: bytes, overwrite: bool = True) -> bool:
        """Atomic write. Returns False (and writes nothing) if overwrite=False and the blob already exists."""
        if self.is_mock:
            path = self._path(blob_path)
            if not overwrite and path.exists():
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            return True
        from azure.core.exceptions import ResourceExistsError
        try:
            self._container().upload_blob(name=self._key(blob_path), data=data, overwrite=overwrite)
            return True
        except ResourceExistsError:
            return False

    def list_blobs(self, prefix: str) -> List[str]:
        """Blob keys starting with prefix (like Azure's name_starts_with)."""
        clean = self._key(prefix)
        if not self.is_mock:
            return sorted(b.name for b in self._container().list_blobs(name_starts_with=clean))
        if not self.base_dir.exists():
            return []
        found = []
        for p in self.base_dir.rglob("*"):
            if p.is_file() and ".tmp" not in p.name:
                rel = p.relative_to(self.base_dir).as_posix()
                if rel.startswith(clean):
                    found.append(rel)
        return sorted(found)

    # -- typed helpers ----------------------------------------------------
    def read_text(self, blob_path: str) -> str:
        return self.read_bytes(blob_path).decode("utf-8")

    def write_text(self, blob_path: str, text: str, overwrite: bool = True) -> bool:
        return self.write_bytes(blob_path, text.encode("utf-8"), overwrite=overwrite)

    def read_json(self, blob_path: str) -> Any:
        return json.loads(self.read_text(blob_path))

    def write_json(self, blob_path: str, data: Any, overwrite: bool = True) -> bool:
        return self.write_text(blob_path, json.dumps(data, indent=2, sort_keys=True, default=str), overwrite=overwrite)

    def write_npz(self, blob_path: str, arrays: Dict[str, Any], overwrite: bool = False) -> bool:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return self.write_bytes(blob_path, buf.getvalue(), overwrite=overwrite)

    def read_npz(self, blob_path: str) -> Dict[str, Any]:
        archive = np.load(io.BytesIO(self.read_bytes(blob_path)), allow_pickle=False)
        return {k: archive[k] for k in archive.files}

    def write_parquet(self, blob_path: str, df, overwrite: bool = True) -> bool:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return self.write_bytes(blob_path, buf.getvalue(), overwrite=overwrite)

    def read_parquet(self, blob_path: str):
        import pandas as pd
        return pd.read_parquet(io.BytesIO(self.read_bytes(blob_path)))


def get_storage_client() -> BlobStorageClient:
    return BlobStorageClient()
