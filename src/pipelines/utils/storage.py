"""
src/pipelines/utils/storage.py
Persistent experiment metadata store using an SQLite EAV model, supporting concurrent
access, partial/exact metadata queries, JSON-to-SQLite migration, and file-path attachment tracking.

## 工作流定位
对应端到端流程第 5 步（训练与存储）。训练脚本通过 create_entry 注册 exp_uuid、写入实验配置
参数（metadata）及文件路径附件（attachments: model、log、eval 目录）。可视化脚本（第 6 步）
通过 read_entries / get_entry 查询 attachments["eval"] 获取 .npy 结果路径。

## 主要组件
- In: 查询包装器，对单一字段做 OR 匹配（field in [v1, v2, ...]）。
- Have: 查询包装器，对列表字段做包含匹配（所有指定值必须在列表中存在）。
- MetadataStorage: 核心存储类。所有公开方法独立获取 SQLite 连接并原子提交；
  WAL 模式允许写操作期间并发读取。

## 输入
- storage_dir（构造参数）：SQLite 数据库及所有附件文件的根目录（默认 "storage/"）。
  数据库路径为 storage_dir/index.db。
- create_entry(metadata, extra_info, attachments, allow_overwrite)：注册新实验。
- update_entry(uuid_query, metadata, extra_info, attachments)：对匹配条目执行字段 upsert。
- read_entries / get_entry(uuid_query 或 metadata_query, exact_match)：查询条目。
- delete_entries：删除数据库行及磁盘上所有关联附件文件。
- cleanup_orphaned_files(clean_entries, clean_unfinished)：清理无有效 DB 条目对应的文件。
- merge_storage(source_storage_dir, on_uuid_conflict, copy_attachments, dry_run)：将另一套 storage
  的实验索引与附件合并到当前 storage。

## 输出 / 副作用
- storage_dir/index.db：SQLite 数据库，含 entries、metadata、extra_info、attachments 四张表。
- update_entry 注册附件时自动创建 storage_dir/ 下的目录和文件（路径无扩展名则视为目录）。
- 首次运行时若 index.json 存在，自动迁移数据到 SQLite 并重命名为 index.json.migrated。
- 每次 read_entries / get_entry 调用都会更新 entries.last_access_time。

## 关键依赖
- 仅使用标准库：sqlite3、json、shutil、pathlib、uuid、datetime。
  fcntl 已保留导入但迁移到 SQLite 后不再使用。

## 注意事项
- _acquire_lock / _release_lock 是保留的空操作桩，仅为兼容检查 _lock_count 的旧测试；
  实际并发安全由 SQLite WAL 模式 + busy_timeout 保障。
- create_entry 的 allow_overwrite: "finished"（默认）仅允许覆盖已完成实验；
  True 允许无条件覆盖；False 禁止任何覆盖。
- _partial_metadata_match 支持将 typing.Any 作为查询值，匹配任意非 None 字段。
- 附件路径以相对字符串存储；传入绝对路径时自动相对于 storage_dir 规范化。
"""
import os
import json
import shutil
import time
import fcntl
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import random
from uuid import uuid4
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = ["MetadataStorage", "In", "Have"]

class In:
    """Wrapper class for multi-value matching in metadata queries."""
    def __init__(self, *values):
        self.values = values
    def __repr__(self):
        return f"In({', '.join(repr(v) for v in self.values)})"
    def __eq__(self, other):
        return other in self.values

class Have:
    """Wrapper class for list element matching in metadata queries."""
    def __init__(self, *values):
        self.values = values
    def __repr__(self):
        return f"Have({', '.join(repr(v) for v in self.values)})"

class MetadataStorage:
    _db_init_locks: Dict[str, threading.Lock] = {}
    _db_init_locks_guard = threading.Lock()

    def __init__(self, storage_dir: Union[str, Path] = "storage"):
        self.storage_dir: Path = Path(storage_dir)
        self.index_file: Path = self.storage_dir / "index.json"
        self.db_file: Path = self.storage_dir / "index.db"
        self.lock_file: Path = self.storage_dir / ".lock"  # Dummy for backward compatibility
        self.index_data: dict = {}  # Dummy for backward compatibility
        self._lock_count: int = 0  # Dummy for backward compatibility

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we need to migrate from JSON
        needs_migration = self.index_file.exists() and not self.db_file.exists()
        
        self._init_database()
        
        if needs_migration:
            logger.info("Detected JSON index, migrating to SQLite...")
            self._migrate_json_to_sqlite()
            migrated_file = self.storage_dir / "index.json.migrated"
            shutil.move(str(self.index_file), str(migrated_file))
            logger.info(f"Migration complete. Old JSON saved as {migrated_file.name}")

    def _init_database(self):
        """Initialize SQLite database with schema"""
        db_key = str(self.db_file.resolve())
        with self._db_init_locks_guard:
            init_lock = self._db_init_locks.setdefault(db_key, threading.Lock())

        with init_lock:
            with self._get_db_connection(configure_wal=True) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entries (
                        uuid TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        last_access_time TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uuid TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        value_type TEXT NOT NULL,
                        FOREIGN KEY (uuid) REFERENCES entries(uuid) ON DELETE CASCADE,
                        UNIQUE(uuid, key)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS extra_info (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uuid TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        value_type TEXT NOT NULL,
                        FOREIGN KEY (uuid) REFERENCES entries(uuid) ON DELETE CASCADE,
                        UNIQUE(uuid, key)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attachments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uuid TEXT NOT NULL,
                        path TEXT NOT NULL,
                        value TEXT NOT NULL,
                        FOREIGN KEY (uuid) REFERENCES entries(uuid) ON DELETE CASCADE
                    )
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_uuid ON metadata(uuid)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata(key)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_value ON metadata(value)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_extra_info_uuid ON extra_info(uuid)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_extra_info_key ON extra_info(key)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attachments_uuid ON attachments(uuid)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_last_access ON entries(last_access_time)")

    def _get_db_connection(self, configure_wal: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_file), timeout=30.0)
        if configure_wal:
            conn.execute("PRAGMA journal_mode=WAL")   # only during serialized initialization
        conn.execute("PRAGMA synchronous=NORMAL")     # faster than FULL; safe against app crash, not OS crash
        conn.execute("PRAGMA foreign_keys=ON")        # enforce ON DELETE CASCADE for child tables
        conn.execute("PRAGMA busy_timeout=30000")     # SQLite-level retry for 30s on locked DB
        return conn

    def _encode_value(self, value: Any) -> Tuple[str, str]:
        if value is None:
            return ('null', 'null')
        elif isinstance(value, bool):
            return (json.dumps(value), 'bool')
        elif isinstance(value, int):
            return (json.dumps(value), 'int')
        elif isinstance(value, float):
            return (json.dumps(value), 'float')
        elif isinstance(value, str):
            return (json.dumps(value), 'str')
        elif isinstance(value, list):
            return (json.dumps(value, ensure_ascii=False), 'list')
        elif isinstance(value, dict):
            return (json.dumps(value, ensure_ascii=False), 'dict')
        else:
            return (json.dumps(str(value)), 'str')

    def _decode_value(self, json_str: str, type_name: str) -> Any:
        if type_name == 'null': return None
        elif type_name in ('bool', 'int', 'float', 'str', 'list', 'dict'):
            return json.loads(json_str)
        else:
            raise ValueError(f"Unknown type: {type_name}")

    def _flatten_attachments(self, attachments: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
        result = []
        for key, value in attachments.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.extend(self._flatten_attachments(value, full_key))
            else:
                result.append((full_key, str(value)))
        return result

    def _unflatten_attachments(self, flat_list: List[Tuple[str, str]]) -> Dict[str, Any]:
        result = {}
        for path, value in flat_list:
            keys = path.split('.')
            current = result
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        return result

    def _normalize_storage_path(self, path: Union[str, Path]) -> Path:
        """Normalize a path to be relative to storage_dir when possible."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            try:
                return path_obj.relative_to(self.storage_dir)
            except ValueError:
                return path_obj
        return path_obj

    def ensure_path(
        self,
        path: Union[str, Path],
        is_file: Optional[bool] = None,
    ) -> Path:
        """Ensure a storage path (or its parent dir) exists."""
        norm_path = self._normalize_storage_path(path)
        full_path = (
            norm_path
            if norm_path.is_absolute()
            else self.storage_dir / norm_path
        )
        if is_file is None:
            is_file = bool(norm_path.suffix)
        if is_file:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            full_path.mkdir(parents=True, exist_ok=True)
        return norm_path

    def ensure_attachment_paths(self, attachments: Dict[str, Any]) -> None:
        """Recursively ensure all attachment paths/directories exist."""
        for value in attachments.values():
            if isinstance(value, dict):
                self.ensure_attachment_paths(value)
                continue
            self.ensure_path(value)

    def _migrate_json_to_sqlite(self):
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON for migration: {e}")
            return

        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            for uuid, entry in index_data.items():
                created_at = entry.get("extra_info", {}).get("created_at", datetime.now().isoformat())
                last_access_time = entry.get("extra_info", {}).get("last_access_time")
                
                cursor.execute(
                    "INSERT OR IGNORE INTO entries (uuid, created_at, last_access_time) VALUES (?, ?, ?)",
                    (uuid, created_at, last_access_time)
                )
                
                metadata = entry.get("metadata", {})
                for key, value in metadata.items():
                    json_value, value_type = self._encode_value(value)
                    cursor.execute(
                        "INSERT OR IGNORE INTO metadata (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                        (uuid, key, json_value, value_type)
                    )
                
                extra_info = entry.get("extra_info", {})
                for key, value in extra_info.items():
                    if key not in ("created_at", "last_access_time"):
                        json_value, value_type = self._encode_value(value)
                        cursor.execute(
                            "INSERT OR IGNORE INTO extra_info (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                            (uuid, key, json_value, value_type)
                        )
                
                attachments = entry.get("attachments", {})
                flat_attachments = self._flatten_attachments(attachments)
                for path, value in flat_attachments:
                    cursor.execute(
                        "INSERT INTO attachments (uuid, path, value) VALUES (?, ?, ?)",
                        (uuid, path, value)
                    )
            logger.info("Migrated JSON to SQLite successfully.")

    # Stub lock methods retained for backward compatibility with tests that inspect _lock_count.
    # Actual concurrency is handled by SQLite WAL + busy_timeout.
    def _acquire_lock(self):
        if not hasattr(self, "_lock_count"):
            self._lock_count = 0
        self._lock_count += 1
    def _release_lock(self):
        if not hasattr(self, "_lock_count"):
            self._lock_count = 0
        if self._lock_count > 0:
            self._lock_count -= 1

    def cleanup_orphaned_files(self, clean_entries: bool = False, clean_unfinished: bool = False):
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT uuid FROM entries")
                all_uuids = [row[0] for row in cursor.fetchall()]
                
                retained_files = set()
                orphaned_index_entries = set()
                
                for u in all_uuids:
                    # check finished
                    cursor.execute("SELECT value, value_type FROM extra_info WHERE uuid = ? AND key = 'finished'", (u,))
                    row = cursor.fetchone()
                    finished = False
                    if row:
                        finished = self._decode_value(row[0], row[1])
                        
                    if not finished and clean_unfinished:
                        orphaned_index_entries.add(u)
                        continue
                        
                    cursor.execute("SELECT path, value FROM attachments WHERE uuid = ?", (u,))
                    att_rows = cursor.fetchall()
                    attachments = self._unflatten_attachments(att_rows)
                    
                    def has_valid_paths(atts: Dict[str, Any]) -> bool:
                        def has_any_file(dir_path: Path) -> bool:
                            try:
                                for item in dir_path.iterdir():
                                    if item.is_file() or (item.is_dir() and has_any_file(item)):
                                        return True
                            except: pass
                            return False
                        for k, v in atts.items():
                            if isinstance(v, dict):
                                if has_valid_paths(v): return True
                            elif isinstance(v, str):
                                path = Path(v)
                                if path.is_absolute():
                                    try: path = path.relative_to(self.storage_dir)
                                    except ValueError: continue
                                full_path = self.storage_dir / path
                                if full_path.is_file() or (full_path.is_dir() and has_any_file(full_path)):
                                    return True
                        return False
                        
                    if not has_valid_paths(attachments):
                        orphaned_index_entries.add(u)
                    
                    def collect_retained_paths(atts: Dict[str, Any]):
                        for k, v in atts.items():
                            if isinstance(v, dict):
                                collect_retained_paths(v)
                            elif isinstance(v, str):
                                path = Path(v)
                                if path.is_absolute():
                                    try: path = path.relative_to(self.storage_dir)
                                    except ValueError: continue
                                full_path = self.storage_dir / path
                                if full_path.is_dir():
                                    try:
                                        for file_path in full_path.rglob('*'):
                                            if file_path.is_file():
                                                retained_files.add(file_path.name)
                                    except: pass
                                else:
                                    retained_files.add(path.name)
                    collect_retained_paths(attachments)
                    
                excluded_files = {self.index_file.name, f"{self.index_file.name}.backup", 
                                  self.db_file.name, f"{self.db_file.name}-wal", f"{self.db_file.name}-shm", ".lock"}
                storage_files = set()
                try:
                    for f in self.storage_dir.iterdir():
                        if f.is_file() and f.name not in excluded_files and not f.name.startswith(".index.json"):
                            storage_files.add(f.name)
                except: pass
                
                orphaned_files = storage_files - retained_files
                for o in orphaned_files:
                    try: (self.storage_dir / o).unlink()
                    except: pass
                
                if clean_entries and orphaned_index_entries:
                    placeholders = ",".join(["?"]*len(orphaned_index_entries))
                    cursor.execute(f"DELETE FROM entries WHERE uuid IN ({placeholders})", list(orphaned_index_entries))

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")

    def merge_storage(
        self,
        source_storage_dir: Union[str, Path],
        on_uuid_conflict: str = "skip",
        copy_attachments: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        Merge entries from another storage directory into the current storage.

        Args:
            source_storage_dir: Source storage root containing index.db and attachments.
            on_uuid_conflict: How to handle UUID collision: "skip", "overwrite", or "error".
            copy_attachments: Whether to copy attachment files/directories from source.
            dry_run: If True, only collect and return merge stats without writing changes.
        """
        if on_uuid_conflict not in {"skip", "overwrite", "error"}:
            raise ValueError("on_uuid_conflict must be one of: 'skip', 'overwrite', 'error'")

        source_storage_path = Path(source_storage_dir)
        if not source_storage_path.exists():
            raise ValueError(f"Source storage directory does not exist: {source_storage_path}")
        if not (source_storage_path / "index.db").exists() and not (source_storage_path / "index.json").exists():
            raise ValueError(
                f"Source storage has no index file (index.db/index.json): {source_storage_path}"
            )

        source_storage = MetadataStorage(source_storage_dir)
        if self.storage_dir.resolve() == source_storage.storage_dir.resolve():
            raise ValueError("Source and destination storage directories are the same")

        stats = {
            "source_entries": 0,
            "inserted_entries": 0,
            "overwritten_entries": 0,
            "skipped_entries": 0,
            "copied_files": 0,
            "copied_dirs": 0,
            "missing_attachments": 0,
            "external_attachments": 0,
        }

        copied_paths = set()

        def iter_attachment_values(obj: Any):
            if isinstance(obj, dict):
                for value in obj.values():
                    yield from iter_attachment_values(value)
            elif isinstance(obj, list):
                for value in obj:
                    yield from iter_attachment_values(value)
            elif isinstance(obj, str):
                yield obj

        with source_storage._get_db_connection() as source_conn, self._get_db_connection() as dest_conn:
            source_entries, source_uuids = source_storage._traverse_entries(_conn=source_conn)
            stats["source_entries"] = len(source_uuids)
            dest_cursor = dest_conn.cursor()

            for entry, uuid in zip(source_entries, source_uuids):
                dest_cursor.execute("SELECT 1 FROM entries WHERE uuid = ?", (uuid,))
                exists = dest_cursor.fetchone() is not None

                if exists:
                    if on_uuid_conflict == "skip":
                        stats["skipped_entries"] += 1
                        continue
                    if on_uuid_conflict == "error":
                        raise ValueError(f"UUID conflict while merging: {uuid}")
                    stats["overwritten_entries"] += 1
                    if not dry_run:
                        dest_cursor.execute("DELETE FROM entries WHERE uuid = ?", (uuid,))
                else:
                    stats["inserted_entries"] += 1

                if not dry_run:
                    created_at = entry.get("extra_info", {}).get("created_at", datetime.now().isoformat())
                    last_access_time = entry.get("extra_info", {}).get("last_access_time")
                    dest_cursor.execute(
                        "INSERT INTO entries (uuid, created_at, last_access_time) VALUES (?, ?, ?)",
                        (uuid, created_at, last_access_time),
                    )

                    for key, value in entry.get("metadata", {}).items():
                        json_value, value_type = self._encode_value(value)
                        dest_cursor.execute(
                            "INSERT OR REPLACE INTO metadata (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                            (uuid, key, json_value, value_type),
                        )

                    for key, value in entry.get("extra_info", {}).items():
                        if key in ("created_at", "last_access_time"):
                            continue
                        json_value, value_type = self._encode_value(value)
                        dest_cursor.execute(
                            "INSERT OR REPLACE INTO extra_info (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                            (uuid, key, json_value, value_type),
                        )

                    flat_attachments = self._flatten_attachments(entry.get("attachments", {}))
                    for path, value in flat_attachments:
                        dest_cursor.execute(
                            "INSERT INTO attachments (uuid, path, value) VALUES (?, ?, ?)",
                            (uuid, path, value),
                        )

                if not copy_attachments:
                    continue

                for attachment_value in iter_attachment_values(entry.get("attachments", {})):
                    path_obj = Path(attachment_value)
                    if path_obj.is_absolute():
                        try:
                            rel_path = path_obj.relative_to(source_storage.storage_dir)
                        except ValueError:
                            stats["external_attachments"] += 1
                            continue
                    else:
                        rel_path = path_obj

                    if rel_path in copied_paths:
                        continue
                    copied_paths.add(rel_path)

                    src_path = source_storage.storage_dir / rel_path
                    dst_path = self.storage_dir / rel_path
                    if not src_path.exists():
                        stats["missing_attachments"] += 1
                        continue
                    if dry_run:
                        continue

                    if src_path.is_dir():
                        dst_path.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        stats["copied_dirs"] += 1
                    else:
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                        stats["copied_files"] += 1

        return stats

    def create_entry(self, uuid: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, extra_info: Optional[Dict[str, Any]] = None, attachments: Optional[Dict[str, Any]] = None, allow_overwrite: Optional[Union[str, bool]] = "finished") -> str:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check for existing entry matching metadata
            if metadata is not None:
                # Scan all entries for exact metadata match to detect duplicates before inserting.
                cursor.execute("SELECT uuid FROM entries")
                all_uuids = [row[0] for row in cursor.fetchall()]
                matched_uuids = []
                for u in all_uuids:
                    # fetch metadata for this uuid
                    cursor.execute("SELECT key, value, value_type FROM metadata WHERE uuid = ?", (u,))
                    meta_dict = {}
                    for k, v, vt in cursor.fetchall():
                        meta_dict[k] = self._decode_value(v, vt)
                    if self._exact_metadata_match(meta_dict, metadata):
                        matched_uuids.append(u)
                        
                if len(matched_uuids) > 1:
                    raise ValueError(f"Multiple entries found with matching metadata: {matched_uuids}")
                elif len(matched_uuids) == 1:
                    u = matched_uuids[0]
                    if allow_overwrite == "finished":
                        cursor.execute("SELECT value, value_type FROM extra_info WHERE uuid = ? AND key = 'finished'", (u,))
                        row = cursor.fetchone()
                        fin = False
                        if row: fin = self._decode_value(row[0], row[1])
                        if not fin:
                            raise ValueError(f"Entry {u} is not finished, cannot overwrite")
                    elif allow_overwrite is False:
                        raise ValueError(f"Entry {u} already exists, cannot overwrite")
                    elif allow_overwrite is True:
                        pass
                    else:
                        raise ValueError(f"Invalid allow_overwrite value: {allow_overwrite}")
                    
                    self.update_entry(uuid_query=u, metadata=metadata, extra_info=extra_info, attachments=attachments, allow_multiple=False, _conn=conn)
                    return u

            if uuid is None:
                while True:
                    test_uuid = str(uuid4())
                    cursor.execute("SELECT 1 FROM entries WHERE uuid = ?", (test_uuid,))
                    if not cursor.fetchone():
                        uuid = test_uuid
                        break
            else:
                cursor.execute("SELECT 1 FROM entries WHERE uuid = ?", (uuid,))
                if cursor.fetchone():
                    raise ValueError(f"Failed to create entry: UUID {uuid} already exists")

            created_at = datetime.now().isoformat()
            cursor.execute("INSERT INTO entries (uuid, created_at, last_access_time) VALUES (?, ?, ?)", (uuid, created_at, created_at))
            
            if metadata or extra_info or attachments:
                self.update_entry(uuid_query=uuid, metadata=metadata, extra_info=extra_info, attachments=attachments, allow_multiple=False, _conn=conn)
            
            return uuid

    def update_entry(self, uuid_query: Optional[Union[List[str], str]] = None, metadata: Optional[Dict[str, Any]] = None, extra_info: Optional[Dict[str, Any]] = None, attachments: Optional[Dict[str, Any]] = None, allow_multiple: bool = False, _conn=None):
        if uuid_query is None and not allow_multiple:
            raise ValueError("Must allow_multiple when uuid_query is None")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        if extra_info is not None and not isinstance(extra_info, dict):
            raise ValueError("Extra information must be a dictionary")
        if attachments is not None and not isinstance(attachments, dict):
            raise ValueError("Attachments must be a dictionary")

        conn = _conn if _conn is not None else self._get_db_connection()
        try:
            cursor = conn.cursor()
            
            if uuid_query is None:
                matched_entries, matched_uuids = self._traverse_entries(metadata_query={}, _conn=conn)
                if len(matched_uuids) == 0:
                    raise ValueError("No entries found")
            else:
                matched_entries, matched_uuids = self._traverse_entries(uuid_query=uuid_query, _conn=conn)
                if len(matched_uuids) == 0:
                    raise ValueError("No entries found")
                if len(matched_uuids) > 1 and not allow_multiple:
                    raise ValueError("Multiple entries found, but allow_multiple is False")
                    
            for u in matched_uuids:
                if metadata:
                    for k, v in metadata.items():
                        json_v, t_v = self._encode_value(v)
                        cursor.execute("INSERT OR REPLACE INTO metadata (uuid, key, value, value_type) VALUES (?, ?, ?, ?)", (u, k, json_v, t_v))
                if extra_info:
                    for k, v in extra_info.items():
                        if k not in ("created_at", "last_access_time"):
                            json_v, t_v = self._encode_value(v)
                            cursor.execute("INSERT OR REPLACE INTO extra_info (uuid, key, value, value_type) VALUES (?, ?, ?, ?)", (u, k, json_v, t_v))
                if attachments:
                    self.ensure_attachment_paths(attachments)
                    flat = self._flatten_attachments(attachments)
                    for path, val in flat:
                        cursor.execute("SELECT id FROM attachments WHERE uuid = ? AND path = ?", (u, path))
                        row = cursor.fetchone()
                        if row:
                            cursor.execute("UPDATE attachments SET value = ? WHERE id = ?", (val, row[0]))
                        else:
                            cursor.execute("INSERT INTO attachments (uuid, path, value) VALUES (?, ?, ?)", (u, path, val))
        finally:
            if _conn is None:
                conn.commit()
                conn.close()

    def append_to_metadata_list(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, key: str = None, values: Union[Any, List[Any]] = None, allow_multiple: bool = False, create_if_missing: bool = True, operation: str = "append"):
        if key is None: raise ValueError("key parameter is required")
        if values is None: raise ValueError("values parameter is required")
        if operation not in ("append", "remove"): raise ValueError("operation must be 'append' or 'remove'")
        if not isinstance(values, list): values = [values]
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            if uuid_query is None and metadata_query is None:
                raise ValueError("Either uuid_query or metadata_query must be provided")
                
            entries, uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, _conn=conn)
            if not uuids:
                raise ValueError("No entries found")
            if len(uuids) > 1 and not allow_multiple:
                raise ValueError(f"Multiple entries found, but allow_multiple is False")
                
            for entry, u in zip(entries, uuids):
                meta = entry["metadata"]
                if key not in meta:
                    if operation == "append" and create_if_missing:
                        meta[key] = []
                    else:
                        raise ValueError(f"Metadata key {key} does not exist")
                if not isinstance(meta[key], list):
                    raise ValueError(f"Metadata key {key} exists but is not a list")
                    
                if operation == "append":
                    for v in values:
                        if v not in meta[key]: meta[key].append(v)
                else:
                    for v in values:
                        if v in meta[key]: meta[key].remove(v)
                        
                json_v, t_v = self._encode_value(meta[key])
                cursor.execute("INSERT OR REPLACE INTO metadata (uuid, key, value, value_type) VALUES (?, ?, ?, ?)", (u, key, json_v, t_v))

    def _traverse_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False, _conn=None) -> Tuple[List[Dict[str, Any]], List[str]]:
        conn = _conn if _conn is not None else self._get_db_connection()
        try:
            cursor = conn.cursor()
            query_uuids = None
            if uuid_query is not None:
                if isinstance(uuid_query, str): query_uuids = [uuid_query]
                else: query_uuids = list(uuid_query)
                placeholders = ",".join(["?"]*len(query_uuids))
                cursor.execute(f"SELECT uuid, created_at, last_access_time FROM entries WHERE uuid IN ({placeholders})", query_uuids)
            else:
                cursor.execute("SELECT uuid, created_at, last_access_time FROM entries")
                
            all_rows = cursor.fetchall()
            filtered_entries = []
            filtered_uuids = []
            
            for u, c_at, last_acc in all_rows:
                cursor.execute("SELECT key, value, value_type FROM metadata WHERE uuid = ?", (u,))
                meta_dict = {}
                for k, v, vt in cursor.fetchall(): meta_dict[k] = self._decode_value(v, vt)
                
                # Check queries BEFORE fetching other heavy data
                match = True
                if metadata_query is not None:
                    if exact_match: match = self._exact_metadata_match(meta_dict, metadata_query)
                    else: match = self._partial_metadata_match(meta_dict, metadata_query)
                    
                if not match: continue
                
                cursor.execute("SELECT key, value, value_type FROM extra_info WHERE uuid = ?", (u,))
                extra_dict = {}
                for k, v, vt in cursor.fetchall(): extra_dict[k] = self._decode_value(v, vt)
                
                if c_at: extra_dict["created_at"] = c_at
                if last_acc: extra_dict["last_access_time"] = last_acc
                
                cursor.execute("SELECT path, value FROM attachments WHERE uuid = ?", (u,))
                att_rows = cursor.fetchall()
                att_dict = {}
                if att_rows: att_dict = self._unflatten_attachments(att_rows)
                
                entry_data = {"uuid": u, "metadata": meta_dict, "extra_info": extra_dict, "attachments": att_dict}
                filtered_entries.append(entry_data)
                filtered_uuids.append(u)
                
            return filtered_entries, filtered_uuids
        finally:
            if _conn is None:
                conn.close()

    def _update_access_time(self, uuids: List[str]):
        current_time = datetime.now().isoformat()
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"]*len(uuids))
            cursor.execute(f"UPDATE entries SET last_access_time = ? WHERE uuid IN ({placeholders})", [current_time] + uuids)

    def read_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not (uuid_query is None or metadata_query is None):
            raise ValueError("Only one can be provided")
            
        entries, uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)
        if uuids:
            self._update_access_time(uuids)
        return entries, uuids

    def get_entry(self, uuid_query: Optional[str] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Dict[str, Any]:
        if not (uuid_query is None or metadata_query is None):
            raise ValueError("Only one can be provided")
        if uuid_query is None and metadata_query is None:
            raise ValueError("One must be provided")
            
        entries, uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)
        if len(entries) != 1:
            raise ValueError(f"{len(entries)} entries found")
        self._update_access_time(uuids)
        return entries[0]

    def _exact_metadata_match(self, file_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        def contains_wrapper(obj):
            if isinstance(obj, (In, Have)): return True
            if isinstance(obj, dict): return any(contains_wrapper(v) for v in obj.values())
            if isinstance(obj, (list, tuple)): return any(contains_wrapper(item) for item in obj)
            return False
        # In/Have wrappers are meaningless in exact-match context; treat as non-match.
        if contains_wrapper(query_metadata): return False
        return file_metadata == query_metadata

    def _partial_metadata_match(self, entry_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        def match(this: Any, query: Any) -> bool:
            if isinstance(query, In):
                if None in query.values and this is None: return True
                return this in query.values
            elif isinstance(query, Have):
                if isinstance(this, list): return all(item in this for item in query.values)
                elif isinstance(this, str): return len(query.values) == 1 and this == query.values[0]
                else: return False
            elif isinstance(query, dict):
                return isinstance(this, dict) and all(match(this.get(key, None), value) for key, value in query.items())
            elif isinstance(query, list):
                return isinstance(this, list) and this == query
            elif query is None: return this is None
            elif query is Any: return this is not None  # typing.Any as sentinel: match any non-None value
            else: return this == query
        return match(entry_metadata, query_metadata)

    def delete_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> int:
        entries, uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)
        if not uuids: return 0
        
        deleted_count = 0
        def recursive_delete(attachments: Dict[str, Any]):
            for v in attachments.values():
                if isinstance(v, dict): recursive_delete(v)
                elif isinstance(v, list):
                    for item in v: recursive_delete(item)
                else:
                    path = self.storage_dir / v
                    if path.is_file(): path.unlink(missing_ok=True)
                    elif path.is_dir(): shutil.rmtree(path, ignore_errors=True)
                    
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            for entry, u in zip(entries, uuids):
                try:
                    recursive_delete(entry.get("attachments", {}))
                    cursor.execute("DELETE FROM entries WHERE uuid = ?", (u,))
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {u}: {e}")
                    
        return deleted_count

    def analyze_metadata_fields(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        entries, uuids = self.read_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)
        if not entries: return {}, {}
        all_fields = set()
        for e in entries: all_fields.update(e.get("metadata", {}).keys())
        
        consistent, inconsistent = {}, {}
        for f in all_fields:
            values = [e.get("metadata", {}).get(f) for e in entries]
            unique = []
            for val in values:
                try:
                    if val not in unique: unique.append(val)
                except TypeError:
                    found = False
                    for ex in unique:
                        if val == ex:
                            found = True; break
                    if not found: unique.append(val)
            if len(unique) == 1: consistent[f] = unique[0]
            else: inconsistent[f] = unique
        return consistent, inconsistent

    def get_storage_stats(self):
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM entries")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT key, COUNT(*) FROM metadata GROUP BY key")
            meta_keys = [f"{r[0]} ({r[1]})" for r in cursor.fetchall()]
            
            # Also prints to stdout to preserve legacy CLI behavior.
            print(f"Storage: {self.storage_dir}")
            print(f"  Total entries: {total}")
            print(f"  Metadata keys: {', '.join(meta_keys)}")
            return f"""Storage: {self.storage_dir}
  Total entries: {total}
  Metadata keys: {', '.join(meta_keys)}"""

if __name__ == "__main__":
    storage = MetadataStorage("data")
    storage.get_storage_stats()
