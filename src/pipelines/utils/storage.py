import os
import json
import shutil
import time
import fcntl
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import random
from uuid import uuid4

logger = logging.getLogger(__name__)

class MetadataStorage:
    """
    JSON-based indexed file storage system with directory-level locking
    
    Features:
    1. Create UUID-named files when storing files and record metadata in index
    2. Query matching files based on metadata
    3. Support partial metadata matching queries
    4. Directory-level locking to prevent concurrent access
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
        self.lock_file: Path = self.storage_dir / ".lock"
        self.index_data: Dict[str, Any] = {}
        self._lock_count: int = 0  # Lock reference count

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _acquire_lock(self):
        """Acquire directory lock by creating a lock file (supports nested locking)"""
        # If lock is already held by this process, just increment count
        if self._lock_count > 0:
            self._lock_count += 1
            logger.debug(f"Lock count incremented to {self._lock_count}")
            return
        
        # First time acquiring lock: actually acquire system lock
        max_retries = 20
        retry_delay = 0.1
        
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
                    retry_delay = min(retry_delay * 2.0, 1.0)
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
                    retry_delay *= 2
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
                # Clean up orphaned files before releasing lock (only on final release)
                self.cleanup_orphaned_files()
                
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
    
    def _load_index(self):
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
    
    def _save_index(self):
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
                self.lock_file.name
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
                            elif isinstance(value, str):
                                path = Path(value)
                                if path.is_absolute():
                                    try:
                                        path = path.relative_to(self.storage_dir)
                                    except ValueError:
                                        continue
                                
                                full_path = self.storage_dir / path
                                if not full_path.exists():
                                    try:
                                        full_path.mkdir(parents=True, exist_ok=True)
                                        logger.debug(f"Created directory: {full_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to create directory {full_path}: {e}")
                                elif full_path.is_file():
                                    pass
                    
                    logger.debug(f"Updated attachments: {attachments}")
                    ensure_directories(attachments)
                    matched_entry["attachments"].update(attachments)
            
            self._save_index()
            logger.debug(f"Updated entry: {uuid_query}")
        finally:
            self._release_lock()
    
    def _traverse_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
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
    
    def read_entries(self, uuid_query: Optional[Union[List[str], str]] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Read entries and their UUIDs based on UUID or metadata query
        
        Args:
            uuid_query: UUID query conditions
            metadata_query: Metadata query conditions
            exact_match: Whether to require exact match, False for partial match
            
        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: Tuple of matching entries and their UUIDs
        """

        if not (uuid_query is None or metadata_query is None):
            raise ValueError("Only one of uuid_query or metadata_query can be provided")
        elif uuid_query is None and metadata_query is None:
            raise ValueError("One of uuid_query or metadata_query must be provided")
        
        try:
            self._acquire_lock()
            
            matched_entries, matched_uuids = self._traverse_entries(uuid_query=uuid_query, metadata_query=metadata_query, exact_match=exact_match)

            logger.debug(f"Found {len(matched_entries)} matching entries for query: {uuid_query if uuid_query is not None else metadata_query}")
            return matched_entries, matched_uuids
        finally:
            self._release_lock()
    
    def get_entry(self, uuid_query: Optional[str] = None, metadata_query: Optional[Dict[str, Any]] = None, exact_match: bool = False) -> Dict[str, Any]:
        """
        Get exactly one entry by UUID or metadata query
        
        Args:
            uuid_query: UUID query conditions
            metadata_query: Metadata query conditions
            exact_match: Whether to require exact match, False for partial match
            
        Returns:
            Dict[str, Any]: Entry data
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
            return matched_entries[0]
        except Exception as e:
            raise ValueError(f"Failed to get entry: {e}")
        finally:
            self._release_lock()
    
    def _exact_metadata_match(self, file_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches exactly"""
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
            if isinstance(query, dict):
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
            metadata_query: Metadata query conditions
            exact_match: Whether to require exact match, False for partial match
            
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
        """Destructor to ensure lock is released"""
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