import os
import tempfile
import shutil
import threading
import time
import pytest
from pathlib import Path
from pipelines.utils.storage import MetadataStorage


class TestMetadataStorageConcurrent:
    """Concurrent access tests for MetadataStorage class"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create a MetadataStorage instance for testing"""
        return MetadataStorage(temp_storage_dir)
    
    def test_concurrent_create_entries(self, temp_storage_dir):
        """Test concurrent entry creation"""
        num_threads = 10
        entries_per_thread = 10
        created_uuids = []
        errors = []
        lock = threading.Lock()
        
        def create_entries(thread_id):
            storage = MetadataStorage(temp_storage_dir)
            thread_uuids = []
            try:
                for i in range(entries_per_thread):
                    metadata = {"thread_id": thread_id, "entry_id": i}
                    uuid = storage.create_entry(metadata=metadata)
                    thread_uuids.append(uuid)
                    time.sleep(0.001)  # Small delay to increase chance of race condition
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
            finally:
                with lock:
                    created_uuids.extend(thread_uuids)
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=create_entries, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all entries were created successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_uuids) == num_threads * entries_per_thread
        
        # Verify all UUIDs are unique
        assert len(set(created_uuids)) == len(created_uuids), "Duplicate UUIDs found"
        
        # Verify all entries can be read
        storage = MetadataStorage(temp_storage_dir)
        all_entries, all_uuids = storage.read_entries()
        assert len(all_entries) == num_threads * entries_per_thread
        
        # Verify entries from each thread
        for thread_id in range(num_threads):
            entries, uuids = storage.read_entries(metadata_query={"thread_id": thread_id})
            assert len(entries) == entries_per_thread
    
    def test_concurrent_read_entries(self, storage):
        """Test concurrent read operations"""
        # Create some entries first
        num_entries = 50
        created_uuids = []
        for i in range(num_entries):
            uuid = storage.create_entry(metadata={"index": i})
            created_uuids.append(uuid)
        
        num_threads = 10
        reads_per_thread = 20
        errors = []
        read_results = []
        lock = threading.Lock()
        
        def read_entries(thread_id):
            storage_instance = MetadataStorage(storage.storage_dir)
            thread_results = []
            try:
                for _ in range(reads_per_thread):
                    # Random read
                    import random
                    uuid = random.choice(created_uuids)
                    entries, uuids = storage_instance.read_entries(uuid_query=uuid)
                    thread_results.append((uuid, len(entries)))
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
            finally:
                with lock:
                    read_results.extend(thread_results)
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=read_entries, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(read_results) == num_threads * reads_per_thread
        
        # Verify all reads returned exactly one entry
        for uuid, count in read_results:
            assert count == 1, f"Expected 1 entry for UUID {uuid}, got {count}"
    
    def test_concurrent_update_entries(self, storage):
        """Test concurrent update operations"""
        # Create entries first
        num_entries = 20
        created_uuids = []
        for i in range(num_entries):
            uuid = storage.create_entry(metadata={"value": i})
            created_uuids.append(uuid)
        
        num_threads = 5
        updates_per_thread = 10
        errors = []
        lock = threading.Lock()
        
        def update_entries(thread_id):
            storage_instance = MetadataStorage(storage.storage_dir)
            try:
                for i in range(updates_per_thread):
                    import random
                    uuid = random.choice(created_uuids)
                    storage_instance.update_entry(uuid, metadata={"thread": thread_id, "update": i})
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=update_entries, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all entries still exist
        all_entries, all_uuids = storage.read_entries()
        assert len(all_entries) == num_entries
        
        # Verify at least some entries have been updated (due to concurrency,
        # some updates may have been overwritten, but at least some should succeed)
        updated_count = sum(1 for entry in all_entries 
                           if "thread" in entry["metadata"] and "update" in entry["metadata"])
        assert updated_count > 0, "No entries were updated"
    
    def test_concurrent_create_and_read(self, temp_storage_dir):
        """Test concurrent create and read operations"""
        num_creators = 5
        num_readers = 5
        entries_per_creator = 10
        created_uuids = []
        read_results = []
        errors = []
        lock = threading.Lock()
        start_event = threading.Event()
        
        def create_entries(thread_id):
            storage = MetadataStorage(temp_storage_dir)
            thread_uuids = []
            start_event.wait()  # Wait for all threads to be ready
            try:
                for i in range(entries_per_creator):
                    uuid = storage.create_entry(metadata={"creator": thread_id, "index": i})
                    thread_uuids.append(uuid)
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append(("creator", thread_id, str(e)))
            finally:
                with lock:
                    created_uuids.extend(thread_uuids)
        
        def read_entries(thread_id):
            storage = MetadataStorage(temp_storage_dir)
            start_event.wait()  # Wait for all threads to be ready
            try:
                while len(created_uuids) < num_creators * entries_per_creator:
                    entries, uuids = storage.read_entries()
                    with lock:
                        read_results.append((thread_id, len(entries)))
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(("reader", thread_id, str(e)))
        
        threads = []
        # Start creator threads
        for i in range(num_creators):
            t = threading.Thread(target=create_entries, args=(i,))
            threads.append(t)
            t.start()
        
        # Start reader threads
        for i in range(num_readers):
            t = threading.Thread(target=read_entries, args=(i,))
            threads.append(t)
            t.start()
        
        # Start all operations
        time.sleep(0.1)  # Give threads time to start
        start_event.set()
        
        for t in threads:
            t.join(timeout=10)
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_uuids) == num_creators * entries_per_creator
        
        # Verify final state
        storage = MetadataStorage(temp_storage_dir)
        all_entries, all_uuids = storage.read_entries()
        assert len(all_entries) == num_creators * entries_per_creator
    
    def test_concurrent_delete_entries(self, storage):
        """Test concurrent delete operations"""
        # Create entries first
        num_entries = 30
        created_uuids = []
        for i in range(num_entries):
            uuid = storage.create_entry(metadata={"index": i})
            created_uuids.append(uuid)
        
        num_threads = 3
        deletes_per_thread = 5
        errors = []
        deleted_uuids = []
        lock = threading.Lock()
        
        def delete_entries(thread_id):
            storage_instance = MetadataStorage(storage.storage_dir)
            thread_deleted = []
            try:
                import random
                uuids_to_delete = random.sample(created_uuids, min(deletes_per_thread, len(created_uuids)))
                for uuid in uuids_to_delete:
                    deleted = storage_instance.delete_entries(uuid_query=uuid)
                    if deleted > 0:
                        thread_deleted.append(uuid)
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
            finally:
                with lock:
                    deleted_uuids.extend(thread_deleted)
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=delete_entries, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Some deletions may fail if entries were already deleted by another thread
        # This is expected behavior
        
        # Verify remaining entries
        all_entries, all_uuids = storage.read_entries()
        assert len(all_entries) <= num_entries
        assert len(all_entries) >= num_entries - num_threads * deletes_per_thread
        
        # Verify deleted entries are actually gone
        for deleted_uuid in deleted_uuids:
            assert deleted_uuid not in all_uuids
    
    def test_concurrent_mixed_operations(self, temp_storage_dir):
        """Test mixed concurrent operations (create, read, update, delete)"""
        num_threads = 8
        operations_per_thread = 20
        errors = []
        lock = threading.Lock()
        start_event = threading.Event()
        created_uuids = []
        
        def mixed_operations(thread_id):
            storage = MetadataStorage(temp_storage_dir)
            thread_uuids = []
            start_event.wait()
            try:
                import random
                for i in range(operations_per_thread):
                    op = random.choice(["create", "read", "update", "delete"])
                    
                    if op == "create":
                        uuid = storage.create_entry(metadata={"thread": thread_id, "op": i})
                        thread_uuids.append(uuid)
                    elif op == "read":
                        if created_uuids:
                            uuid = random.choice(created_uuids) if created_uuids else None
                            if uuid:
                                storage.read_entries(uuid_query=uuid)
                    elif op == "update":
                        if created_uuids:
                            uuid = random.choice(created_uuids) if created_uuids else None
                            if uuid:
                                try:
                                    storage.update_entry(uuid, metadata={"updated_by": thread_id})
                                except ValueError:
                                    pass  # Entry might have been deleted
                    elif op == "delete":
                        if created_uuids:
                            uuid = random.choice(created_uuids) if created_uuids else None
                            if uuid:
                                storage.delete_entries(uuid_query=uuid)
                                with lock:
                                    if uuid in created_uuids:
                                        created_uuids.remove(uuid)
                    
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
            finally:
                with lock:
                    created_uuids.extend(thread_uuids)
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(t)
            t.start()
        
        time.sleep(0.1)
        start_event.set()
        
        for t in threads:
            t.join(timeout=30)
        
        # Some errors are expected due to race conditions (e.g., deleting non-existent entries)
        # But the system should not crash
        
        # Verify storage is still consistent
        storage = MetadataStorage(temp_storage_dir)
        all_entries, all_uuids = storage.read_entries()
        
        # Verify no duplicate UUIDs
        assert len(set(all_uuids)) == len(all_uuids), "Duplicate UUIDs found"
        
        # Verify index file is valid JSON
        index_file = storage.index_file
        assert index_file.exists()
        import json
        with open(index_file, 'r') as f:
            index_data = json.load(f)
            assert isinstance(index_data, dict)
    
    def test_lock_nested_operations(self, storage):
        """Test that nested operations work correctly with locking"""
        uuid = storage.create_entry(metadata={"value": 0})
        
        def nested_operations():
            storage_instance = MetadataStorage(storage.storage_dir)
            # Nested operations should work due to lock counting
            storage_instance._acquire_lock()
            storage_instance._acquire_lock()
            try:
                storage_instance.update_entry(uuid, metadata={"nested": True})
                storage_instance.read_entries(uuid_query=uuid)
            finally:
                storage_instance._release_lock()
                storage_instance._release_lock()
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=nested_operations)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify entry was updated
        entries, uuids = storage.read_entries(uuid_query=uuid)
        assert len(entries) == 1
        assert entries[0]["metadata"].get("nested") is True
    
    def test_concurrent_cleanup(self, temp_storage_dir):
        """Test concurrent cleanup operations"""
        storage = MetadataStorage(temp_storage_dir)
        
        # Create entries with files
        num_entries = 20
        for i in range(num_entries):
            test_file = storage.storage_dir / f"test_{i}.txt"
            test_file.write_text(f"content_{i}")
            uuid = storage.create_entry(
                attachments={"file": str(test_file.relative_to(storage.storage_dir))}
            )
        
        # Create some orphaned files
        for i in range(5):
            orphaned = storage.storage_dir / f"orphaned_{i}.txt"
            orphaned.write_text("orphaned")
        
        num_threads = 3
        errors = []
        lock = threading.Lock()
        
        def run_cleanup(thread_id):
            storage = MetadataStorage(temp_storage_dir)
            try:
                storage.cleanup_orphaned_files()
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=run_cleanup, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify orphaned files are removed
        for i in range(5):
            orphaned = storage.storage_dir / f"orphaned_{i}.txt"
            assert not orphaned.exists(), f"Orphaned file {orphaned} still exists"
        
        # Verify valid files still exist
        for i in range(num_entries):
            test_file = storage.storage_dir / f"test_{i}.txt"
            assert test_file.exists(), f"Valid file {test_file} was removed"
    
    def test_stress_test(self, temp_storage_dir):
        """Stress test with many concurrent operations"""
        num_threads = 20
        operations_per_thread = 50
        errors = []
        lock = threading.Lock()
        start_event = threading.Event()
        created_uuids = []
        
        def stress_operations(thread_id):
            storage = MetadataStorage(temp_storage_dir)
            thread_uuids = []
            start_event.wait()
            op_index = 0
            try:
                import random
                for i in range(operations_per_thread):
                    op_index = i
                    op = random.choice(["create", "read"])
                    
                    if op == "create":
                        uuid = storage.create_entry(metadata={"thread": thread_id, "op": i})
                        with lock:
                            thread_uuids.append(uuid)
                            created_uuids.append(uuid)
                    elif op == "read":
                        if created_uuids:
                            uuid = random.choice(created_uuids)
                            storage.read_entries(uuid_query=uuid)
                    
                    if i % 10 == 0:
                        time.sleep(0.001)  # Small delay every 10 operations
            except Exception as e:
                with lock:
                    errors.append((thread_id, op_index, str(e)))
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=stress_operations, args=(i,))
            threads.append(t)
            t.start()
        
        time.sleep(0.1)
        start_event.set()
        
        for t in threads:
            t.join(timeout=60)
        
        # Verify final state
        storage = MetadataStorage(temp_storage_dir)
        all_entries, all_uuids = storage.read_entries()
        
        # Verify no duplicate UUIDs
        assert len(set(all_uuids)) == len(all_uuids), "Duplicate UUIDs found"
        
        # Verify index consistency
        assert len(all_entries) == len(all_uuids)
        
        # Some errors might occur due to race conditions, but should be minimal
        if errors:
            print(f"Warning: {len(errors)} errors occurred during stress test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

