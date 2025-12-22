#!/usr/bin/env python3
"""
Test for async_train.py scheduler.
"""
import pytest
import tempfile
import shutil
import time
import subprocess
import signal
import os
import pathlib
import re
from pathlib import Path


class TestAsyncTrain:
    """Test suite for async_train.py"""
    
    @pytest.fixture
    def temp_test_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_jobs_dir(self, temp_test_dir):
        """Create jobs directory"""
        jobs_dir = Path(temp_test_dir) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        return jobs_dir
    
    @pytest.fixture
    def test_processed_dir(self, temp_test_dir):
        """Create processed directory"""
        processed_dir = Path(temp_test_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir
    
    @pytest.fixture
    def test_logs_dir(self, temp_test_dir):
        """Create logs directory"""
        logs_dir = Path(temp_test_dir) / "logs" / "history"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def generate_job_file(self, jobs_dir, task_durations, id_bias=0, file_name="test_job.txt", project_root=None, memory_percent=30.0):
        """Generate a job file with tasks"""
        job_file = jobs_dir / file_name
        with job_file.open("w") as f:
            for i, duration in enumerate(task_durations):
                task_id = f"task_{i + id_bias}"
                if project_root:
                    script_path = Path(project_root) / "scripts" / "test_gpu_work.py"
                    task_cmd = f"uv run {script_path} --duration {duration} --task-id {task_id} --memory-percent {memory_percent}"
                else:
                    task_cmd = f"uv run scripts/test_gpu_work.py --duration {duration} --task-id {task_id} --memory-percent {memory_percent}"
                f.write(f"{task_cmd}\n")
        return job_file
    
    def parse_log_file(self, log_file):
        """Parse log file to extract task information"""
        tasks_info = {}
        if not log_file.exists():
            return tasks_info
        
        with log_file.open("r") as f:
            for line in f:
                # Look for task start messages
                # Format: <asctime> - TASK_task_task_X - INFO - Starting task task_X, duration: X.XXs
                # The logger name is task_{task_id}, so if task_id is "task_2", logger name becomes "task_task_2"
                match = re.search(r'TASK_task_(task_\d+) - INFO - Starting task (task_\d+), duration: ([\d.]+)s', line)
                if match:
                    task_id = match.group(2)  # Use the task_id from the message, not the logger name
                    duration = float(match.group(3))
                    if task_id not in tasks_info:
                        tasks_info[task_id] = {
                            'start_time': None,
                            'end_time': None,
                            'duration': duration,
                            'completed': False
                        }
                
                # Look for task completion messages
                # Format: <asctime> - TASK_task_task_X - INFO - Task task_X completed successfully after X.XXs
                match = re.search(r'TASK_task_(task_\d+) - INFO - Task (task_\d+) completed successfully after ([\d.]+)s', line)
                if match:
                    task_id = match.group(2)  # Use the task_id from the message
                    elapsed = float(match.group(3))
                    if task_id not in tasks_info:
                        tasks_info[task_id] = {
                            'start_time': None,
                            'end_time': None,
                            'duration': None,
                            'completed': False
                        }
                    tasks_info[task_id]['completed'] = True
                    tasks_info[task_id]['elapsed'] = elapsed
        
        return tasks_info
    
    def check_job_file_status(self, job_file):
        """Check the status of tasks in job file"""
        statuses = {}
        if not job_file.exists():
            return statuses
        
        with job_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("#"):
                    task = line[1:].strip()
                    statuses[task] = "finished"
                elif line.startswith("!"):
                    task = line[1:].strip()
                    statuses[task] = "running"
                elif line.startswith("?"):
                    task = line[1:].strip()
                    statuses[task] = "terminated"
                else:
                    statuses[line] = "waiting"
        
        return statuses
    
    def test_async_train_basic(self, temp_test_dir, test_jobs_dir, test_processed_dir, test_logs_dir):
        """Test basic functionality of async_train with 8 GPUs"""
        import random
        
        # Save original working directory
        original_cwd = os.getcwd()
        
        # Generate tasks for 8 GPUs, total runtime around 60 seconds
        # Each task runs 5-10 seconds, with 8 GPUs parallel, we need about 12-16 tasks
        # Expected total sequential time: 12 * 7.5 = 90 seconds, but with 8 GPUs parallel: ~90/8 = ~11 seconds
        # To get ~60 seconds total, we need more tasks: 60 * 8 / 7.5 = ~64 tasks
        # Actually, let's aim for ~60 seconds of work distributed across 8 GPUs
        # If each task is 5-10 seconds, and we want ~60 seconds total runtime with 8 GPUs:
        # We need enough tasks so that sequential time is ~60 * 8 = 480 seconds
        # With 5-10 second tasks, that's about 48-96 tasks. Let's use 60 tasks.
        num_tasks = 60
        task_durations = [random.uniform(5.0, 10.0) for _ in range(num_tasks)]
        expected_total_sequential_time = sum(task_durations)
        # With 8 GPUs, expected parallel time is approximately sequential_time / 8
        expected_parallel_time = expected_total_sequential_time / 8.0
        
        # Generate job file (use default "jobs" directory name)
        jobs_dir = Path(temp_test_dir) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_file = self.generate_job_file(jobs_dir, task_durations, project_root=original_cwd, memory_percent=30.0)
        
        # Create processed directory (default name)
        processed_dir = Path(temp_test_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create available_gpus.txt - use 8 GPUs (0-7)
        gpu_file = Path(temp_test_dir) / "available_gpus.txt"
        gpu_file.write_text("0 1 2 3 4 5 6 7")
        
        # Change to temp directory
        proc = None
        try:
            os.chdir(temp_test_dir)
            
            # Start async_train as subprocess with max_memory=50, grace_period=5
            script_path = Path(original_cwd) / "scripts" / "async_train.py"
            proc = subprocess.Popen(
                ["uv", "run", "python", str(script_path), "--max-memory", "50", "--grace-period", "5"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_test_dir
            )
            
            # Wait for about 70 seconds (60 seconds work + 10 seconds buffer)
            wait_time = 70
            print(f"Waiting {wait_time} seconds for tasks to complete...")
            
            start_time = time.time()
            try:
                # Wait for process or timeout
                proc.wait(timeout=wait_time)
            except subprocess.TimeoutExpired:
                # If timeout, that's okay - we'll check results and then terminate
                pass
            
            # Give it a bit more time to finish current tasks
            time.sleep(3)
            
            # Terminate the process
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            
            actual_runtime = time.time() - start_time
            
            # Check results (log file is in logs/history/history.log)
            log_file = Path(temp_test_dir) / "logs" / "history" / "history.log"
            tasks_info = self.parse_log_file(log_file)
            
            # Check job file status
            job_statuses = self.check_job_file_status(job_file)
            
            # Verify results
            completed_tasks = [tid for tid, info in tasks_info.items() if info.get('completed', False)]
            finished_in_file = [task for task, status in job_statuses.items() if status == "finished"]
            
            # Calculate expected total sequential time from completed tasks
            completed_sequential_time = sum(
                info.get('duration', 0) for tid, info in tasks_info.items() 
                if info.get('completed', False)
            )
            
            # Calculate ratio
            if completed_sequential_time > 0:
                time_ratio = actual_runtime / (completed_sequential_time / 8.0)
            else:
                time_ratio = float('inf')
            
            print(f"\nTest Results:")
            print(f"Total tasks: {num_tasks}")
            print(f"Expected total sequential time: {expected_total_sequential_time:.2f}s")
            print(f"Expected parallel time (8 GPUs): {expected_parallel_time:.2f}s")
            print(f"Completed tasks sequential time: {completed_sequential_time:.2f}s")
            print(f"Expected parallel time for completed: {completed_sequential_time / 8.0:.2f}s")
            print(f"Actual runtime: {actual_runtime:.2f}s")
            print(f"Time ratio (actual / expected_parallel): {time_ratio:.2f}")
            print(f"Tasks completed (from logs): {len(completed_tasks)}/{num_tasks}")
            print(f"Tasks finished (from job file): {len(finished_in_file)}/{num_tasks}")
            
            # Verify at least some tasks completed
            assert len(completed_tasks) > 0, "No tasks completed"
            
            # Verify log file exists and has content
            assert log_file.exists(), "Log file should exist"
            
            # Report the time ratio (no assertion, just report)
            print(f"\nPerformance Summary:")
            print(f"  Actual runtime: {actual_runtime:.2f}s")
            print(f"  Expected parallel time: {completed_sequential_time / 8.0:.2f}s")
            print(f"  Ratio: {time_ratio:.2f}x")
            
        finally:
            os.chdir(original_cwd)
            # Cleanup process if still running
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.kill()
                except:
                    pass
    
    def test_async_train_multiple_job_files(self, temp_test_dir, test_jobs_dir, test_processed_dir, test_logs_dir):
        """Test async_train with multiple job files on 8 GPUs"""
        import random
        
        # Change to temp directory
        original_cwd = os.getcwd()
        
        # Generate multiple job files (use default "jobs" directory name)
        jobs_dir = Path(temp_test_dir) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        
        num_files = 3
        tasks_per_file = 20  # More tasks for 8 GPUs
        
        job_files = []
        all_durations = []
        
        for file_idx in range(num_files):
            durations = [random.uniform(5.0, 10.0) for _ in range(tasks_per_file)]
            all_durations.extend(durations)
            job_file = self.generate_job_file(jobs_dir, durations, id_bias=file_idx * tasks_per_file, file_name=f"job_{file_idx}.txt", project_root=original_cwd, memory_percent=30.0)
            job_files.append(job_file)
        
        expected_total_sequential_time = sum(all_durations)
        expected_parallel_time = expected_total_sequential_time / 8.0
        
        # Create processed directory (default name)
        processed_dir = Path(temp_test_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create available_gpus.txt - use 8 GPUs
        gpu_file = Path(temp_test_dir) / "available_gpus.txt"
        gpu_file.write_text("0 1 2 3 4 5 6 7")
        
        proc = None
        try:
            os.chdir(temp_test_dir)
            
            # Start async_train with max_memory=50, grace_period=5
            script_path = Path(original_cwd) / "scripts" / "async_train.py"
            proc = subprocess.Popen(
                ["uv", "run", "python", str(script_path), "--max-memory", "50", "--grace-period", "5"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_test_dir
            )
            
            # Wait for some tasks to complete
            wait_time = expected_parallel_time * 2.0  # Wait up to 200% of expected parallel time
            print(f"Waiting {wait_time:.2f} seconds...")
            
            start_time = time.time()
            try:
                proc.wait(timeout=wait_time)
            except subprocess.TimeoutExpired:
                pass
            
            time.sleep(3)
            
            # Terminate
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            
            actual_runtime = time.time() - start_time
            
            # Check results (log file is in logs/history/history.log)
            log_file = Path(temp_test_dir) / "logs" / "history" / "history.log"
            tasks_info = self.parse_log_file(log_file)
            
            completed_tasks = [tid for tid, info in tasks_info.items() if info.get('completed', False)]
            
            # Calculate expected total sequential time from completed tasks
            completed_sequential_time = sum(
                info.get('duration', 0) for tid, info in tasks_info.items() 
                if info.get('completed', False)
            )
            
            # Calculate ratio
            if completed_sequential_time > 0:
                time_ratio = actual_runtime / (completed_sequential_time / 8.0)
            else:
                time_ratio = float('inf')
            
            print(f"\nTest Results:")
            print(f"Total tasks: {len(all_durations)}")
            print(f"Expected total sequential time: {expected_total_sequential_time:.2f}s")
            print(f"Expected parallel time (8 GPUs): {expected_parallel_time:.2f}s")
            print(f"Completed tasks sequential time: {completed_sequential_time:.2f}s")
            print(f"Expected parallel time for completed: {completed_sequential_time / 8.0:.2f}s")
            print(f"Actual runtime: {actual_runtime:.2f}s")
            print(f"Time ratio (actual / expected_parallel): {time_ratio:.2f}")
            print(f"Tasks completed: {len(completed_tasks)}/{len(all_durations)}")
            print(f"Task info: {tasks_info}")
            
            # Verify at least some tasks completed
            assert len(completed_tasks) > 0, "No tasks completed"
            
            # Verify log file exists and has content
            assert log_file.exists(), "Log file should exist"
            
            # Verify all job files were processed
            for job_file in job_files:
                job_statuses = self.check_job_file_status(job_file)
                assert len(job_statuses) > 0, f"Job file {job_file} was not processed"
            
            # Report the time ratio
            print(f"\nPerformance Summary:")
            print(f"  Actual runtime: {actual_runtime:.2f}s")
            print(f"  Expected parallel time: {completed_sequential_time / 8.0:.2f}s")
            print(f"  Ratio: {time_ratio:.2f}x")
            
        finally:
            os.chdir(original_cwd)
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

