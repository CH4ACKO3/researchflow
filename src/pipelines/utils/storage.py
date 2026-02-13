import os
import json
import shutil
import time
import fcntl
import sqlite3
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
    """
    Wrapper class for multi-value matching in metadata queries.
    
    Matches if the entry's field value is in the provided list of values.
    Special handling for None: if None is in values, also matches entries where the field doesn't exist.
    
    Example:
        # Match entries where seed is 0, 1, 2, or 3
        metadata_query = {"seed": In(0, 1, 2, 3)}
        
        # Match entries where seed doesn't exist (None) or equals 42
        metadata_query = {"seed": In(None, 42)}
    """
    
    def __init__(self, *values):
        """
        Initialize In wrapper with values to match against.
        
        Args:
            *values: Variable number of values to match. Can include None to match missing fields.
        """
        self.values = values
    
    def __repr__(self):
        return f"In({', '.join(repr(v) for v in self.values)})"
    
    def __eq__(self, other):
        """Check if other value is in the values list"""
        return other in self.values


class Have:
    """
    Wrapper class for list element matching in metadata queries.
    
    Matches if the entry's field value is a list containing at least one of the provided values.
    
    Example:
        # Match entries where tags list contains "tag1"
        metadata_query = {"tags": Have("tag1")}
        
        # Match entries where tags list contains "tag1" or "tag2"
        metadata_query = {"tags": Have("tag1", "tag2")}
    """
    
    def __init__(self, *values):
        """
        Initialize Have wrapper with values to match against.
        
        Args:
            *values: Variable number of values to check for in the list.
        """
        self.values = values
    
    def __repr__(self):
        return f"Have({', '.join(repr(v) for v in self.values)})"

class MetadataStorage:
    """
    SQLite-based indexed file storage system with directory-level locking
    
    Features:
    1. Create UUID-named files when storing files and record metadata in SQLite database
    2. Query matching files based on metadata using efficient SQL queries
    3. Support partial metadata matching queries with In wrapper for multi-value matching
    4. Directory-level locking to prevent concurrent access
    5. Automatic tracking of last access time for each entry
    6. WAL mode for improved concurrency
    
    Storage Backend:
        Uses SQLite database (index.db) with EAV (Entity-Attribute-Value) model
        for flexible metadata storage. Automatically migrates from legacy JSON
        format if index.json is found.
    
    Multi-value matching:
        Use the In wrapper to match entries where a field value is in a list of values:
        
        Example:
            # Match entries where seed is 0, 1, 2, or 3
            entries, uuids = storage.read_entries(metadata_query={"seed": In(0, 1, 2, 3)})
            
            # Match entries where seed doesn't exist (None) or equals 42
            entries, uuids = storage.read_entries(metadata_query={"seed": In(None, 42)})
    
    List element matching:
        Use the Have wrapper to match entries where a field's list value contains at least one of the specified elements:
        
        Example:
            # Match entries where tags list contains "tag1"
            entries, uuids = storage.read_entries(metadata_query={"tags": Have("tag1")})
            
            # Match entries where tags list contains "tag1" or "tag2"
            entries, uuids = storage.read_entries(metadata_query={"tags": Have("tag1", "tag2")})
    
    Access time tracking:
        The system automatically records the last access time in extra_info["last_access_time"]
        when entries are read via read_entries() or get_entry(). The timestamp is in ISO format.
        This enables future cleanup based on access patterns.
    """
    
    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize storage system
        
        Args:
            storage_dir: File storage directory
            index_file: Index file name
        """
        self.storage_dir: Path = Path(storage_dir)
        self.index_file: Path = self.storage_dir / "index.json"
        self.index_file_backup: Path = self.storage_dir / "index.json.backup"
        self.db_file: Path = self.storage_dir / "index.db"
        self.lock_file: Path = self.storage_dir / ".lock"
        self.index_data: Dict[str, Any] = {}
        self._lock_count: int = 0  # Lock reference count
        self.use_sqlite: bool = True  # Use SQLite backend by default

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and handle migration
        if self.use_sqlite:
            # Check if we need to migrate from JSON
            if self.index_file.exists() and not self.db_file.exists():
                logger.info("Detected JSON index, migrating to SQLite...")
                # Load JSON data first
                try:
                    with open(self.index_file, 'r', encoding='utf-8') as f:
                        self.index_data = json.load(f)
                    logger.info(f"Loaded {len(self.index_data)} entries from JSON")
                except Exception as e:
                    logger.error(f"Failed to load JSON for migration: {e}")
                    self.index_data = {}
                
                # Initialize database and migrate
                self._init_database()
                if self.index_data:
                    self._migrate_json_to_sqlite()
                    # Rename old JSON file to backup
                    migrated_file = self.storage_dir / "index.json.migrated"
                    shutil.move(self.index_file, migrated_file)
                    logger.info(f"Migration complete. Old JSON saved as {migrated_file.name}")
            else:
                # Just initialize database
                self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Create entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    uuid TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_access_time TEXT
                )
            """)
            
            # Create metadata table (EAV model)
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
            
            # Create extra_info table (EAV model)
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
            
            # Create attachments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    path TEXT NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (uuid) REFERENCES entries(uuid) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_uuid ON metadata(uuid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata(key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_value ON metadata(value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extra_info_uuid ON extra_info(uuid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extra_info_key ON extra_info(key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attachments_uuid ON attachments(uuid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entries_last_access ON entries(last_access_time)")
            
            conn.commit()
            logger.debug("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with WAL mode enabled
        
        Returns:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(str(self.db_file))
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    
    def _encode_value(self, value: Any) -> Tuple[str, str]:
        """
        Encode a value to JSON string and determine its type
        
        Args:
            value: Value to encode
            
        Returns:
            Tuple[str, str]: (json_encoded_string, type_name)
        """
        if value is None:
            return ('null', 'null')
        elif isinstance(value, bool):  # Must check before int
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
            # Handle Path and other types by converting to string
            return (json.dumps(str(value)), 'str')
    
    def _decode_value(self, json_str: str, type_name: str) -> Any:
        """
        Decode a JSON string back to its original type
        
        Args:
            json_str: JSON encoded string
            type_name: Type name ('str', 'int', 'float', 'bool', 'null', 'list', 'dict')
            
        Returns:
            Decoded value
        """
        if type_name == 'null':
            return None
        elif type_name == 'bool':
            return json.loads(json_str)
        elif type_name == 'int':
            return json.loads(json_str)
        elif type_name == 'float':
            return json.loads(json_str)
        elif type_name == 'str':
            return json.loads(json_str)
        elif type_name == 'list':
            return json.loads(json_str)
        elif type_name == 'dict':
            return json.loads(json_str)
        else:
            raise ValueError(f"Unknown type: {type_name}")
    
    def _flatten_attachments(self, attachments: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
        """
        Flatten nested attachments dictionary into list of (path, value) tuples
        
        Args:
            attachments: Nested attachments dictionary
            prefix: Path prefix for recursion
            
        Returns:
            List of (dotted_path, value) tuples
            
        Example:
            {"output": {"videos": "path/to/videos", "images": "path/to/images"}}
            -> [("output.videos", "path/to/videos"), ("output.images", "path/to/images")]
        """
        result = []
        for key, value in attachments.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                # Recurse into nested dict
                result.extend(self._flatten_attachments(value, full_key))
            elif isinstance(value, (str, Path)):
                # Leaf value - store as string
                result.append((full_key, str(value)))
            else:
                # Other types - convert to string
                result.append((full_key, str(value)))
        return result
    
    def _unflatten_attachments(self, flat_list: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Unflatten list of (path, value) tuples back to nested dictionary
        
        Args:
            flat_list: List of (dotted_path, value) tuples
            
        Returns:
            Nested attachments dictionary
            
        Example:
            [("output.videos", "path/to/videos"), ("output.images", "path/to/images")]
            -> {"output": {"videos": "path/to/videos", "images": "path/to/images"}}
        """
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
    
    def _migrate_json_to_sqlite(self):
        """
        Migrate data from index.json to SQLite database
        
        This method reads the current index_data and inserts all entries
        into the SQLite database.
        """
        if not self.index_data:
            logger.debug("No data to migrate from JSON to SQLite")
            return
        
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            
            for uuid, entry in self.index_data.items():
                # Insert into entries table
                created_at = entry.get("extra_info", {}).get("created_at", datetime.now().isoformat())
                last_access_time = entry.get("extra_info", {}).get("last_access_time")
                
                cursor.execute(
                    "INSERT OR REPLACE INTO entries (uuid, created_at, last_access_time) VALUES (?, ?, ?)",
                    (uuid, created_at, last_access_time)
                )
                
                # Insert metadata
                metadata = entry.get("metadata", {})
                for key, value in metadata.items():
                    json_value, value_type = self._encode_value(value)
                    cursor.execute(
                        "INSERT OR REPLACE INTO metadata (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                        (uuid, key, json_value, value_type)
                    )
                
                # Insert extra_info (excluding last_access_time which is in entries table)
                extra_info = entry.get("extra_info", {})
                for key, value in extra_info.items():
                    if key not in ("created_at", "last_access_time"):
                        json_value, value_type = self._encode_value(value)
                        cursor.execute(
                            "INSERT OR REPLACE INTO extra_info (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                            (uuid, key, json_value, value_type)
                        )
                
                # Insert attachments
                attachments = entry.get("attachments", {})
                flat_attachments = self._flatten_attachments(attachments)
                for path, value in flat_attachments:
                    cursor.execute(
                        "INSERT INTO attachments (uuid, path, value) VALUES (?, ?, ?)",
                        (uuid, path, value)
                    )
            
            conn.commit()
            logger.info(f"Migrated {len(self.index_data)} entries from JSON to SQLite")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to migrate JSON to SQLite: {e}")
            raise
        finally:
            conn.close()
    
    def _migrate_sqlite_to_json(self):
        """
        Migrate data from SQLite database to index.json format
        
        This method reads all data from SQLite and reconstructs the
        index_data dictionary.
        """
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Get all entries
            cursor.execute("SELECT uuid, created_at, last_access_time FROM entries")
            entries = cursor.fetchall()
            
            self.index_data = {}
            
            for uuid, created_at, last_access_time in entries:
                entry = {
                    "uuid": uuid,
                    "metadata": {},
                    "extra_info": {},
                    "attachments": {}
                }
                
                # Get metadata
                cursor.execute("SELECT key, value, value_type FROM metadata WHERE uuid = ?", (uuid,))
                for key, value, value_type in cursor.fetchall():
                    entry["metadata"][key] = self._decode_value(value, value_type)
                
                # Get extra_info
                cursor.execute("SELECT key, value, value_type FROM extra_info WHERE uuid = ?", (uuid,))
                for key, value, value_type in cursor.fetchall():
                    entry["extra_info"][key] = self._decode_value(value, value_type)
                
                # Add timestamps from entries table only if they were originally in extra_info
                # We check if created_at was stored in extra_info table
                cursor.execute("SELECT key FROM extra_info WHERE uuid = ? AND key = 'created_at'", (uuid,))
                if cursor.fetchone():
                    if created_at:
                        entry["extra_info"]["created_at"] = created_at
                        
                if last_access_time:
                    entry["extra_info"]["last_access_time"] = last_access_time
                
                # Get attachments
                cursor.execute("SELECT path, value FROM attachments WHERE uuid = ?", (uuid,))
                flat_attachments = cursor.fetchall()
                if flat_attachments:
                    entry["attachments"] = self._unflatten_attachments(flat_attachments)
                
                self.index_data[uuid] = entry
            
            logger.info(f"Migrated {len(self.index_data)} entries from SQLite to JSON")
        except Exception as e:
            logger.error(f"Failed to migrate SQLite to JSON: {e}")
            raise
        finally:
            conn.close()
    
    def _acquire_lock(self):
        """Acquire directory lock by creating a lock file (supports nested locking)"""
        # If lock is already held by this process, just increment count
        if self._lock_count > 0:
            self._lock_count += 1
            logger.debug(f"Lock count incremented to {self._lock_count}")
            return
        
        # First time acquiring lock: actually acquire system lock
        max_retries = 50
        retry_delay = 0.01  # Start with 10ms
        
        for attempt in range(max_retries):
            try:
                # Create lock file with exclusive access
                # Don't use 'with' statement as we need to keep the file handle open
                f = open(self.lock_file, 'w')
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Keep file handle open to maintain lock
                self._lock_handle = f
                self._lock_count = 1
                logger.debug("Directory lock acquired")
                self._load_index()
                return
            except BlockingIOError:
                # Lock is held by another process
                if attempt < max_retries - 1:
                    # Add jitter (randomization) to avoid thundering herd problem
                    jitter = random.uniform(0, retry_delay * 0.5)
                    actual_delay = retry_delay + jitter
                    logger.debug(f"Directory is locked, retrying in {actual_delay:.3f}s (attempt {attempt + 1})")
                    time.sleep(actual_delay)
                    retry_delay = min(retry_delay * 1.5, 0.2)  # Max 200ms delay
                else:
                    logger.error(f"Failed to acquire directory lock after {max_retries} attempts")
                    raise RuntimeError("Failed to acquire directory lock")
            except Exception as e:
                if attempt < max_retries - 1:
                    # Add jitter for other exceptions as well
                    jitter = random.uniform(0, retry_delay * 0.5)
                    actual_delay = retry_delay + jitter
                    logger.warning(f"Failed to acquire lock (attempt {attempt + 1}): {e}")
                    time.sleep(actual_delay)
                    retry_delay = min(retry_delay * 1.5, 0.2)
                else:
                    logger.error(f"Failed to acquire lock after {max_retries} attempts: {e}")
                    raise
    
    def _release_lock(self):
        """Release directory lock (supports nested locking)"""
        # If lock count > 1, just decrement count
        if self._lock_count > 1:
            self._lock_count -= 1
            logger.debug(f"Lock count decremented to {self._lock_count}")
            return
        
        # Last release: actually release system lock
        if self._lock_count == 0:
            logger.warning("Attempted to release lock that was not acquired")
            return
        
        try:
            if hasattr(self, '_lock_handle') and self._lock_handle:
                # Note: Removed automatic cleanup on lock release to avoid race conditions
                # Users should call cleanup_orphaned_files() explicitly when needed
                
                # First release the lock, then close the file
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
                self._lock_count = 0
                logger.debug("Directory lock released")
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            # Try to clean up the handle even if lock release failed
            try:
                if hasattr(self, '_lock_handle') and self._lock_handle:
                    self._lock_handle.close()
                    self._lock_handle = None
                self._lock_count = 0
            except:
                pass
    
    def _load_index_json(self):
        """Load index file with fallback to backup if corrupted"""
        # Try loading main index file
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index_data = json.load(f)
                    logger.debug(f"Index file loaded, containing {len(self.index_data)} records")
                return
            except Exception as e:
                logger.error(f"Failed to load index file: {e}")
                # Try loading backup
                if self.index_file_backup.exists():
                    try:
                        with open(self.index_file_backup, 'r', encoding='utf-8') as f:
                            self.index_data = json.load(f)
                            logger.warning(f"Loaded index from backup, containing {len(self.index_data)} records")
                            # Restore backup to main file
                            shutil.copy2(self.index_file_backup, self.index_file)
                            logger.info("Restored index file from backup")
                        return
                    except Exception as backup_error:
                        logger.error(f"Failed to load backup index file: {backup_error}")
                
                # Both main and backup failed, start fresh
                logger.warning("Both index and backup failed to load, creating new index")
                self.index_data = {}
        else:
            logger.debug("Index file does not exist, creating new index")
            self.index_data = {}
    
    def _load_index_sqlite(self):
        """Load data from SQLite database to memory (for compatibility period)"""
        # Ensure database is initialized before loading
        if not self.db_file.exists():
            logger.debug("Database doesn't exist yet, starting with empty index")
            self.index_data = {}
            return
        self._migrate_sqlite_to_json()
    
    def _load_index(self):
        """
        Load index data from storage backend
        
        Routes to appropriate backend based on use_sqlite flag
        """
        if self.use_sqlite:
            self._load_index_sqlite()
        else:
            self._load_index_json()
    
    def _save_index_json(self):
        """Save index file atomically with backup"""
        temp_file = self.storage_dir / f".index.json.tmp.{uuid4()}"
        try:
            # Custom encoder to handle Path objects and other non-serializable types
            class PathEncoder(json.JSONEncoder):
                def default(self, o):
                    # Handle Path and its subclasses (PosixPath, WindowsPath, etc.)
                    if isinstance(o, Path):
                        return str(o)
                    # Let the base class raise TypeError for other non-serializable types
                    return super().default(o)
            
            # Write to temporary file first
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_data, f, ensure_ascii=False, indent=2, cls=PathEncoder)
                # Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())
            
            # Backup current index file if it exists
            if self.index_file.exists():
                shutil.copy2(self.index_file, self.index_file_backup)
            
            # Atomically replace the index file
            # os.replace() is atomic on both Unix and Windows
            os.replace(temp_file, self.index_file)
            
            logger.debug("Index file saved successfully")
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            logger.error(f"Failed to save index file: {e}")
            raise
    
    def _save_index_sqlite(self):
        """Save data from memory to SQLite database"""
        # For now, we use a simple approach: clear and reinsert all data
        # This ensures consistency but could be optimized later
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Get existing UUIDs in database
            cursor.execute("SELECT uuid FROM entries")
            db_uuids = {row[0] for row in cursor.fetchall()}
            
            # Get UUIDs in memory
            mem_uuids = set(self.index_data.keys())
            
            # Delete entries that are no longer in memory
            deleted_uuids = db_uuids - mem_uuids
            for uuid in deleted_uuids:
                cursor.execute("DELETE FROM entries WHERE uuid = ?", (uuid,))
            
            # Insert or update entries that are in memory
            for uuid, entry in self.index_data.items():
                # Check if entry exists
                cursor.execute("SELECT uuid FROM entries WHERE uuid = ?", (uuid,))
                exists = cursor.fetchone() is not None
                
                # Update entries table
                extra_info = entry.get("extra_info", {})
                created_at = extra_info.get("created_at", datetime.now().isoformat())
                last_access_time = extra_info.get("last_access_time")
                
                if exists:
                    cursor.execute(
                        "UPDATE entries SET last_access_time = ? WHERE uuid = ?",
                        (last_access_time, uuid)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO entries (uuid, created_at, last_access_time) VALUES (?, ?, ?)",
                        (uuid, created_at, last_access_time)
                    )
                
                # Update metadata (delete and reinsert for simplicity)
                cursor.execute("DELETE FROM metadata WHERE uuid = ?", (uuid,))
                metadata = entry.get("metadata", {})
                for key, value in metadata.items():
                    json_value, value_type = self._encode_value(value)
                    cursor.execute(
                        "INSERT INTO metadata (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                        (uuid, key, json_value, value_type)
                    )
                
                # Update extra_info (delete and reinsert)
                cursor.execute("DELETE FROM extra_info WHERE uuid = ?", (uuid,))
                extra_info = entry.get("extra_info", {})
                for key, value in extra_info.items():
                    if key not in ("created_at", "last_access_time"):
                        json_value, value_type = self._encode_value(value)
                        cursor.execute(
                            "INSERT INTO extra_info (uuid, key, value, value_type) VALUES (?, ?, ?, ?)",
                            (uuid, key, json_value, value_type)
                        )
                
                # Update attachments (delete and reinsert)
                cursor.execute("DELETE FROM attachments WHERE uuid = ?", (uuid,))
                attachments = entry.get("attachments", {})
                flat_attachments = self._flatten_attachments(attachments)
                for path, value in flat_attachments:
                    cursor.execute(
                        "INSERT INTO attachments (uuid, path, value) VALUES (?, ?, ?)",
                        (uuid, path, value)
                    )
            
            conn.commit()
            logger.debug(f"Saved {len(self.index_data)} entries to SQLite")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save to SQLite: {e}")
            raise
        finally:
            conn.close()
    
    def _save_index(self):
        """
        Save index data to storage backend
        
        Routes to appropriate backend based on use_sqlite flag
        """
        if self.use_sqlite:
            self._save_index_sqlite()
        else:
            self._save_index_json()
    
    def cleanup_orphaned_files(self, clean_entries: bool = False, clean_unfinished: bool = False):
        """        
        Args:
            clean_entries: If True, also remove index entries that have no attachments
            clean_unfinished: If True, also remove index entries that are not finished
        """
        try:
            self._acquire_lock()
            # Collect all files that should be retained based on attachments
            retained_files = set()
            orphaned_index_entries = set()

            def collect_retained_paths(attachments: Dict[str, Any]):
                """Recursively collect all file and directory paths that should be retained"""
                for key, value in attachments.items():
                    if isinstance(value, dict):
                        # Nested dictionary, recurse
                        collect_retained_paths(value)
                    elif isinstance(value, str):
                        # This is a path (file or directory)
                        path = Path(value)
                        if path.is_absolute():
                            # If it's an absolute path, check if it's within storage_dir
                            try:
                                path = path.relative_to(self.storage_dir)
                            except ValueError:
                                # Path is outside storage_dir, skip
                                continue

                        if (self.storage_dir / path).is_dir():
                            # Directory path: retain all files in this directory and subdirectories
                            try:
                                for file_path in (self.storage_dir / path).rglob('*'):
                                    if file_path.is_file():
                                        retained_files.add(file_path.name)
                            except Exception as e:
                                logger.warning(f"Failed to collect files from directory {path}: {e}")
                        else:
                            # File path: retain this specific file
                            retained_files.add(path.name)
                    # Skip other types (lists, etc.) for now

            def has_valid_paths(attachments: Dict[str, Any]) -> bool:
                """Check if attachments contain any valid file or directory paths"""
                def has_any_file(dir_path: Path) -> bool:
                    """Recursively check if directory contains at least one file"""
                    try:
                        for item in dir_path.iterdir():
                            if item.is_file():
                                return True
                            elif item.is_dir():
                                if has_any_file(item):
                                    return True
                    except Exception:
                        pass
                    return False
                
                for key, value in attachments.items():
                    if isinstance(value, dict):
                        # Nested dictionary, recurse
                        if has_valid_paths(value):
                            return True
                    elif isinstance(value, str):
                        # This is a path (file or directory)
                        path = Path(value)
                        if path.is_absolute():
                            # If it's an absolute path, check if it's within storage_dir
                            try:
                                path = path.relative_to(self.storage_dir)
                            except ValueError:
                                # Path is outside storage_dir, skip
                                continue

                        full_path = self.storage_dir / path
                        if full_path.is_file():
                            # File exists
                            return True
                        elif full_path.is_dir():
                            # Directory exists and contains at least one file
                            if has_any_file(full_path):
                                return True
                return False

            # Collect retained files from all index entries' attachments
            for uuid_val, entry_info in list(self.index_data.items()):
                attachments = entry_info.get("attachments", {})
                collect_retained_paths(attachments)
                # Check if entry should be marked as orphaned
                if entry_info.get("extra_info", {}).get("finished", False) == False and clean_unfinished:
                    orphaned_index_entries.add(uuid_val)
                    continue
                if not has_valid_paths(attachments):
                    orphaned_index_entries.add(uuid_val)

            # Get all files currently in storage directory (excluding system files)
            excluded_files = {
                self.index_file.name,
                self.index_file_backup.name,
                self.lock_file.name,
                self.db_file.name,  # Exclude SQLite database
                f"{self.db_file.name}-wal",  # WAL file
                f"{self.db_file.name}-shm",  # Shared memory file
            }
            storage_files = set()
            for file_path in self.storage_dir.iterdir():
                # Exclude system files and temporary files
                if file_path.is_file() and file_path.name not in excluded_files and not file_path.name.startswith('.index.json.tmp'):
                    storage_files.add(file_path.name)

            # Find orphaned files (files in storage but not in retained set)
            orphaned_files = storage_files - retained_files

            # Remove orphaned files
            for orphaned_file in orphaned_files:
                file_path = self.storage_dir / orphaned_file
                try:
                    file_path.unlink()
                    logger.debug(f"Removed orphaned file: {orphaned_file}")
                except Exception as e:
                    logger.error(f"Failed to remove orphaned file {orphaned_file}: {e}")

            # Remove orphaned index entries
            if clean_entries:
                self.delete_entries(uuid_query=list(orphaned_index_entries))

            # Save index if any changes were made
            if orphaned_files or (clean_entries and orphaned_index_entries):
                self._save_index()
                logger.debug(f"Cleanup completed: removed {len(orphaned_files)} orphaned files and {len(orphaned_index_entries)} orphaned index entries")
            else:
                logger.debug("No cleanup needed - all files and index entries are consistent")

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
        finally:
            self._release_lock()
    
    def create_entry(self, uuid: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, extra_info: Optional[Dict[str, Any]] = None, attachments: Optional[Dict[str, Any]] = None, allow_overwrite: Optional[Union[str, bool]] = "finished") -> str:
        """
        Create or update an entry
        
        If metadata is provided, checks for existing entries with exact metadata match:
        - 0 matches: create new entry
        - 1 match: overwrite existing entry (use its UUID)
        - >1 matches: raise error
        
        Args:
            uuid: UUID of the entry; if None, a new UUID will be generated
        
        Optional arguments:
            metadata: Metadata of the entry
            extra_info: Extra information of the entry
            attachments: Attachments of the entry

        Returns:
            str: UUID of the created/updated entry
        """
        try:
            self._acquire_lock()
            
            # Check for existing entry with exact metadata match
            if metadata is not None:
                matched_entries, matched_uuids = self._traverse_entries(metadata_query=metadata, exact_match=True)
                
                if len(matched_uuids) > 1:
                    raise ValueError(f"Multiple entries found with matching metadata: {matched_uuids}")
                elif len(matched_uuids) == 1:
                    # Overwrite existing entry
                    if allow_overwrite == "finished":
                        if not self.index_data[matched_uuids[0]].get("extra_info", {}).get("finished", False) == True:
                            raise ValueError(f"Entry {matched_uuids[0]} is not finished, cannot overwrite")
                    elif allow_overwrite == True:
                        pass
                    elif allow_overwrite == False:
                        raise ValueError(f"Entry {matched_uuids[0]} already exists, cannot overwrite")
                    else:
                        raise ValueError(f"Invalid allow_overwrite value: {allow_overwrite}")

                    uuid = matched_uuids[0]
                    logger.debug(f"Found existing entry with matching metadata, overwriting: {uuid}")
                    self.update_entry(uuid, metadata, extra_info, attachments)
                    return uuid
            
            # Create new entry
            if uuid is None:
                while not uuid or uuid in self.index_data:
                    uuid = str(uuid4())
            else:
                assert uuid not in self.index_data, f"UUID {uuid} already exists"

            self.index_data[uuid] = {
                "uuid": uuid,
                "metadata": {},
                "extra_info": {},
                "attachments": {}
            }
            
            self._save_index()

            if any([metadata, extra_info, attachments]):
                self.update_entry(uuid, metadata, extra_info, attachments)
            
            logger.debug(f"Created entry: {uuid}")
        except Exception as e:
            if uuid and uuid in self.index_data:
                self.delete_entries(uuid_query=uuid)
            raise ValueError(f"Failed to create entry: {e}")
        finally:
            self._release_lock()

        return uuid

    def update_entry(self, uuid_query: Optional[Union[List[str], str]] = None, metadata: Optional[Dict[str, Any]] = None, extra_info: Optional[Dict[str, Any]] = None, attachments: Optional[Dict[str, Any]] = None, allow_multiple: bool = False):
        """
        Update entry and record metadata
        
        Args:
            uuid: UUID of the entry to update, if None, update all entries
            metadata: Entry metadata information
            extra_info: Extra information to store with the entry
            attachments: Attachments to store with the entry
            allow_multiple: If True, allow updating multiple entries
        Returns:
            str: UUID of updated entry
        """
        if uuid_query is None and not allow_multiple:
            raise ValueError("Must allow_multiple when uuid_query is None")
        
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"Metadata must be a dictionary: {metadata}")
        if extra_info is not None and not isinstance(extra_info, dict):
            raise ValueError(f"Extra information must be a dictionary: {extra_info}")
        if attachments is not None and not isinstance(attachments, dict):
            raise ValueError(f"Attachments must be a dictionary: {attachments}")
        
        try:
            self._acquire_lock()
            
            if uuid_query is None:
                matched_entries, matched_uuids = self._traverse_entries(metadata_query={})
                if len(matched_entries) == 0:
                    raise ValueError(f"No entries found for metadata: {metadata}")
            else:
                matched_entries, matched_uuids = self._traverse_entries(uuid_query=uuid_query)
                if len(matched_entries) == 0:
                    raise ValueError(f"No entries found for UUID: {uuid_query}")
                if len(matched_entries) > 1 and not allow_multiple:
                    raise ValueError(f"Multiple entries found for UUID: {uuid_query}, but allow_multiple is False")
                
            for matched_entry, matched_uuid in zip(matched_entries, matched_uuids):
                if metadata is not None:
                    matched_entry["metadata"].update(metadata)
                if extra_info is not None:
                    matched_entry["extra_info"].update(extra_info)
                if attachments is not None:
                    # Ensure directories in attachments exist
                    def ensure_directories(attachments: Dict[str, Any]):
                        """Recursively ensure directories in attachments exist"""
                        for key, value in attachments.items():
                            logger.debug(f"Checking attachment directory: {key} -> {value}")
                            if isinstance(value, dict):
                                # Nested dictionary, recurse
                                ensure_directories(value)
                            elif isinstance(value, str) or isinstance(value, Path):
                                if isinstance(value, str):
                                    try:
                                        value = Path(value)
                                    except Exception as e:
                                        raise ValueError(f"Invalid attachment path: {value}: {e}")
                                path = value
                                if path.is_absolute():
                                    try:
                                        path = path.relative_to(self.storage_dir)
                                    except ValueError:
                                        continue
                                full_path = self.storage_dir / path
                                
                                if not full_path.exists():
                                    if '.' in str(path):
                                        # Likely a file, create parent directory
                                        parent_dir = full_path.parent
                                        if not parent_dir.exists():
                                            try:
                                                parent_dir.mkdir(parents=True, exist_ok=True)
                                                logger.debug(f"Created parent directory: {parent_dir}")
                                            except Exception as e:
                                                logger.warning(f"Failed to create parent directory {parent_dir}: {e}")
                                    else:
                                        # Likely a directory, create it
                                        try:
                                            full_path.mkdir(parents=True, exist_ok=True)
                                            logger.debug(f"Created directory: {full_path}")
                                        except Exception as e:
                                            logger.warning(f"Failed to create directory {full_path}: {e}")
                            else:
                                raise ValueError(f"Invalid attachment type: {type(value)}")
                    
                    logger.debug(f"Updated attachments: {attachments}")
                    ensure_directories(attachments)
                    matched_entry["attachments"].update(attachments)
            
            self._save_index()
            logger.debug(f"Updated entry: {uuid_query}")
        finally:
            self._release_lock()
    
    def _traverse_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Internal method to traverse and match entries based on UUID or metadata queries (legacy JSON backend)
        
        Args:
            uuid_query: UUID or list of UUIDs to match
            metadata_query: Metadata conditions to match (supports In wrapper for multi-value matching and Have wrapper for list element matching)
            exact_match: If True, requires exact metadata match; if False, allows partial match
            
        Returns:
            Tuple of (matched entries list, matched UUIDs list)
        """
        if uuid_query is None and metadata_query is None:
            return list(self.index_data.values()), list(self.index_data.keys())
        
        matched_entires: list[Dict[str, Any]] = []
        matched_uuids: list[str] = []
        
        if uuid_query is not None:
            uuid_query = uuid_query if isinstance(uuid_query, list) else [uuid_query]
                
            for entry_uuid, entry_data in self.index_data.items():
                if entry_uuid in uuid_query:
                    matched_entires.append(entry_data)
                    matched_uuids.append(entry_uuid)
            
        elif metadata_query is not None:
            for entry_uuid, entry_data in self.index_data.items():
                entry_metadata: Dict[Any, Any] = entry_data.get("metadata", {})
                
                if exact_match:
                    if self._exact_metadata_match(entry_metadata, metadata_query):
                        matched_entires.append(entry_data)
                        matched_uuids.append(entry_uuid)
                else:
                    if self._partial_metadata_match(entry_metadata, metadata_query):
                        matched_entires.append(entry_data)
                        matched_uuids.append(entry_uuid)

        return matched_entires, matched_uuids
    
    def _update_access_time(self, uuids: List[str]):
        """
        Update last access time for given entries
        
        Args:
            uuids: List of UUIDs to update access time for
        """
        current_time = datetime.now().isoformat()
        for uuid in uuids:
            if uuid in self.index_data:
                if "extra_info" not in self.index_data[uuid]:
                    self.index_data[uuid]["extra_info"] = {}
                self.index_data[uuid]["extra_info"]["last_access_time"] = current_time
        
        # Save index after updating access times
        self._save_index()
    
    def read_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Read entries and their UUIDs based on UUID or metadata query
        
        Args:
            uuid_query: UUID query conditions
            metadata_query: Metadata query conditions. Supports In wrapper for multi-value matching.
            exact_match: Whether to require exact match, False for partial match.
                        Note: In wrapper is only supported when exact_match=False.
            
        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: Tuple of matching entries and their UUIDs
            
        Side effects:
            Updates extra_info["last_access_time"] for all matched entries with current timestamp
            
        Examples:
            # Match entries where seed is 0, 1, 2, or 3
            entries, uuids = storage.read_entries(metadata_query={"seed": In(0, 1, 2, 3)})
            
            # Match entries where seed doesn't exist or equals 42
            entries, uuids = storage.read_entries(metadata_query={"seed": In(None, 42)})
            
            # Match entries where tags list contains "tag1"
            entries, uuids = storage.read_entries(metadata_query={"tags": Have("tag1")})
            
            # Read all entries (no query parameters)
            entries, uuids = storage.read_entries()
        """

        if not (uuid_query is None or metadata_query is None):
            raise ValueError("Only one of uuid_query or metadata_query can be provided")
        
        try:
            self._acquire_lock()
            
            matched_entries, matched_uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)

            # Update last access time for matched entries
            if matched_uuids:
                self._update_access_time(matched_uuids)

            logger.debug(f"Found {len(matched_entries)} matching entries for query: {uuid_query if uuid_query is not None else metadata_query}")
            return matched_entries, matched_uuids
        finally:
            self._release_lock()
    
    def get_entry(self, uuid_query: Optional[str] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Dict[str, Any]:
        """
        Get exactly one entry by UUID or metadata query
        
        Args:
            uuid_query: UUID query conditions
            metadata_query: Metadata query conditions. Supports In wrapper for multi-value matching and Have wrapper for list element matching.
            exact_match: Whether to require exact match, False for partial match.
                        Note: In and Have wrappers are only supported when exact_match=False.
            
        Returns:
            Dict[str, Any]: Entry data
            
        Side effects:
            Updates extra_info["last_access_time"] for the matched entry with current timestamp
            
        Raises:
            ValueError: If number of matching entries is not exactly 1
        """
        if not (uuid_query is None or metadata_query is None):
            raise ValueError("Only one of uuid_query or metadata_query can be provided")
        elif uuid_query is None and metadata_query is None:
            raise ValueError("One of uuid_query or metadata_query must be provided")

        try:
            self._acquire_lock()
            matched_entries, matched_uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)

            if len(matched_entries) != 1:
                raise ValueError(f"{len(matched_entries)} entries found for query: {uuid_query if uuid_query is not None else metadata_query}")
            
            # Update last access time for the matched entry
            self._update_access_time(matched_uuids)
            
            return matched_entries[0]
        except Exception as e:
            raise ValueError(f"Failed to get entry: {e}")
        finally:
            self._release_lock()
    
    def _exact_metadata_match(self, file_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        """
        Check if metadata matches exactly
        
        Args:
            file_metadata: Metadata of the file entry
            query_metadata: Metadata query conditions for exact match
            
        Returns:
            bool: True if metadata matches exactly, False otherwise
            
        Note:
            In and Have wrappers are not supported in exact match mode and will be treated as not matching.
        """
        # Check if query contains In or Have wrapper (not supported in exact match)
        def contains_wrapper(obj: Any) -> bool:
            if isinstance(obj, (In, Have)):
                return True
            elif isinstance(obj, dict):
                return any(contains_wrapper(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return any(contains_wrapper(item) for item in obj)
            return False
        
        if contains_wrapper(query_metadata):
            logger.warning("In or Have wrapper detected in exact_match mode, this will not match")
            return False
            
        return file_metadata == query_metadata
    
    def _partial_metadata_match(self, entry_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        """
        Check if metadata matches partially
        
        Args:
            file_metadata: Metadata of the file
            query_metadata: Metadata query conditions
            
        Returns:
            bool: True if metadata matches partially, False otherwise
        """
        def match(this: Any, query: Any) -> bool:
            # Handle In wrapper for multi-value matching
            if isinstance(query, In):
                # If None is in values and field doesn't exist (this is None), it's a match
                if None in query.values and this is None:
                    return True
                # Otherwise check if value is in the list
                return this in query.values
            # Handle Have wrapper for list element matching
            elif isinstance(query, Have):
                # Check if this is a list and contains at least one of the query values
                if not isinstance(this, list):
                    return False
                return any(item in this for item in query.values)
            elif isinstance(query, dict):
                return isinstance(this, dict) and all(match(this.get(key, None), value) for key, value in query.items())
            elif isinstance(query, list):
                return isinstance(this, list) and this == query
            elif query is None:
                return this is None
            elif query is Any:
                return this is not None
            else:
                return this == query

        return match(entry_metadata, query_metadata)
    
    def delete_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> int:
        """
        Delete files based on metadata query
        
        Args:
            uuid_query: UUID query conditions
            metadata_query: Metadata query conditions. Supports In wrapper for multi-value matching and Have wrapper for list element matching.
            exact_match: Whether to require exact match, False for partial match.
                        Note: In and Have wrappers are only supported when exact_match=False.
            
        Returns:
            int: Number of files deleted
        """
        try:
            self._acquire_lock()
            
            # Find matching files
            matched_entries, matched_uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)
            
            if not matched_uuids:
                logger.debug(f"No files found matching metadata: {metadata_query}")
                return 0
            
            deleted_count = 0

            def recursive_delete(attachments: Dict[str, Any]):
                for attachment in attachments.values():
                    if isinstance(attachment, dict):
                        recursive_delete(attachment)
                    elif isinstance(attachment, list):
                        for item in attachment:
                            recursive_delete(item)
                    else:
                        path = self.storage_dir / attachment
                        if path.is_file():
                            path.unlink(missing_ok=True)
                        elif path.is_dir():
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            # Path doesn't exist, safe to ignore
                            pass
            
            for matched_entry, matched_uuid in zip(matched_entries, matched_uuids):
                attachments: Dict[str, Any] = matched_entry.get("attachments", {})
                try:
                    recursive_delete(attachments)
                    logger.debug(f"Deleted attachments: {attachments}")
                    del self.index_data[matched_uuid]
                    logger.debug(f"Deleted entry: {matched_uuid}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete entry {matched_uuid}: {e}")
            
            # Save index if any files were deleted
            if deleted_count > 0:
                self._save_index()
                logger.info(f"Deleted {deleted_count} files matching metadata: {metadata_query}")
            
            return deleted_count
            
        finally:
            self._release_lock()
    
    def analyze_metadata_fields(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[List[str], Dict[str, List[Any]]]:
        """
        Analyze metadata fields across queried entries to identify consistent and inconsistent fields
        
        Args:
            uuid_query: UUID or list of UUIDs to query
            metadata_query: Metadata conditions to match (supports In wrapper for multi-value matching and Have wrapper for list element matching)
            exact_match: If True, requires exact metadata match; if False, allows partial match
            
        Returns:
            Tuple of:
            - consistent_fields: List of field names that have the same value across all entries
            - inconsistent_fields: Dict mapping field names to list of all possible values (including None)
                                   Format: {"seed": [None, 0, 1, 42], "model": ["resnet", "vgg"], ...}
        """
        try:
            self._acquire_lock()
            
            # Query entries using the same parameters as read_entries
            entries, uuids = self.read_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)
            
            if not entries:
                logger.debug("No entries found for analysis")
                return [], {}
            
            # Collect all field names across all entries
            all_fields = set()
            for entry in entries:
                metadata = entry.get("metadata", {})
                all_fields.update(metadata.keys())
            
            # Analyze each field
            consistent_fields = []
            inconsistent_fields = {}
            
            for field in all_fields:
                # Collect all values for this field (None if field doesn't exist in an entry)
                values = []
                for entry in entries:
                    metadata = entry.get("metadata", {})
                    value = metadata.get(field)
                    values.append(value)
                
                # Get unique values
                unique_values = []
                for val in values:
                    # Check if value already in unique_values (handle unhashable types)
                    try:
                        if val not in unique_values:
                            unique_values.append(val)
                    except TypeError:
                        # For unhashable types like lists/dicts, do manual comparison
                        found = False
                        for existing_val in unique_values:
                            if val == existing_val:
                                found = True
                                break
                        if not found:
                            unique_values.append(val)
                
                # Determine if field is consistent or inconsistent
                if len(unique_values) == 1:
                    # All entries have the same value for this field
                    consistent_fields.append(field)
                else:
                    # Field has different values across entries
                    inconsistent_fields[field] = unique_values
            
            logger.debug(f"Analyzed {len(entries)} entries: {len(consistent_fields)} consistent fields, {len(inconsistent_fields)} inconsistent fields")
            return consistent_fields, inconsistent_fields
            
        finally:
            self._release_lock()
    
    def get_storage_stats(self):
        """
        Print storage statistics
        """
        try:
            self._acquire_lock()
            
            total_entries = len(self.index_data)
            
            # Count metadata key usage frequency
            metadata_keys = {}
            for entry_info in self.index_data.values():
                for key in entry_info.get("metadata", {}).keys():
                    metadata_keys[key] = metadata_keys.get(key, 0) + 1
            
            print(f"Storage: {self.storage_dir}")
            print(f"  Total entries: {total_entries}")
            print(f"  Metadata keys: {', '.join(metadata_keys.keys())}")
        finally:
            self._release_lock()

    def __del__(self):
        """
        Destructor to ensure lock is released when object is garbage collected
        
        Attempts to release any held locks to prevent resource leaks.
        Errors are silently ignored during cleanup to avoid issues in object destruction.
        """
        try:
            if hasattr(self, '_lock_handle') and self._lock_handle:
                self._release_lock()
        except:
            pass  # Ignore errors during cleanup
    

if __name__ == "__main__":
    storage = MetadataStorage("data")
    print(storage.get_storage_stats())
    storage = MetadataStorage("model")
    print(storage.get_storage_stats())