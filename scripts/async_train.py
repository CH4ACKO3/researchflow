import asyncio
import sys
import os
import fileinput
import time
import datetime
import subprocess
import pynvml
import logging, logging.handlers
import argparse
from collections import defaultdict, deque
import pathlib

def get_system_memory_usage():
    """Get system memory usage percentage"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            lines = meminfo.split('\n')
            mem_total = 0
            mem_available = 0
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1])
                elif line.startswith('MemFree:'):
                    if mem_available == 0:
                        mem_available = int(line.split()[1])
            
            if mem_total > 0:
                mem_used = mem_total - mem_available
                return (mem_used / mem_total) * 100.0
            return 0.0
    except Exception:
        return 0.0
    
# Initialize NVIDIA Management Library
pynvml.nvmlInit()

# Initialize logging
history_logger = logging.getLogger()
history_logger.setLevel(logging.INFO)
log_file_path = pathlib.Path("logs/history/history.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.handlers.TimedRotatingFileHandler(str(log_file_path), when="midnight", backupCount=7)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
history_logger.addHandler(file_handler)

console_logger = logging.getLogger("console")
console_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
console_logger.addHandler(console_handler)
console_logger.propagate = False

LOG_FILE = log_file_path.open("a")

# Global gpu status
available_gpus = []
gpus_lock = asyncio.Lock()

def parse_task(line):
    if line.startswith("!"):
        return line[1:].strip(), "running"
    elif line.startswith("#"):
        return line[1:].strip(), "finished"
    elif line.startswith("?"):
        return line[1:].strip(), "terminated"
    else:
        return line.strip(), "waiting"

class TaskIO:
    def __init__(self, jobs_dir="jobs", processed_dir="processed"):
        self.jobs_dir = pathlib.Path(jobs_dir)
        self.processed_dir = pathlib.Path(processed_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup async data structures
        self.task_queue = asyncio.Queue()
        self.message_queue = asyncio.Queue()
        self.task_pool = defaultdict(set)
        self.task_pool_lock = asyncio.Lock()
        self.file_with_changes = set()
        self.file_last_mtime = defaultdict(float)  # Key: Path object
        self.file_io_lock = asyncio.Lock()
        
        self.scan_switch = True
        self.scan_switch_condition = asyncio.Condition()

        self.should_stop = False
    
    async def start(self):
        self.scan_file_loop = asyncio.create_task(self.scan_file())
        self.monitor_file_loop = asyncio.create_task(self.monitor_file())
        self.task_manage_loop = asyncio.create_task(self.task_manage())

    async def stop(self):
        await asyncio.gather(self.monitor_file_loop, self.task_manage_loop, self.scan_file_loop, return_exceptions=True)
    
    def query_pool_status(self, task):
        for status, task_set in self.task_pool.items():
            if task in task_set:
                return status
        return None

    def change_pool_status(self, task, target_status):
        for status, task_set in self.task_pool.items():
            if task in task_set:
                task_set.discard(task)
                break
        self.task_pool[target_status].add(task)

    def task_to_line(self, task, status, first_line_flag):
        if status == "finished":
            return f"{'\n' if not first_line_flag else ''}# {task}"
        elif status == "terminated":
            return f"{'\n' if not first_line_flag else ''}? {task}"
        elif status == "running":
            return f"{'\n' if not first_line_flag else ''}! {task}"
        else:
            return f"{'\n' if not first_line_flag else ''}{task}"

    async def monitor_file(self):
        try:
            while True:
                grace_time = asyncio.create_task(asyncio.sleep(1))
                any_change = False
                async with self.file_io_lock:
                    for file_path in self.jobs_dir.rglob("*.txt"):
                        if file_path.stat().st_mtime > self.file_last_mtime[file_path]:
                            # self.file_last_mtime[file_path] = file_path.stat().st_mtime
                            any_change = True
                                
                if any_change:
                    async with self.scan_switch_condition:
                        self.scan_switch = True
                        self.scan_switch_condition.notify_all()
                
                await grace_time
        except asyncio.CancelledError:
            pass

    async def task_manage(self):
        try:
            while True:
                message = await self.message_queue.get()
                async with self.task_pool_lock:
                    task, status = message
                    self.change_pool_status(task, status)
                
                async with self.scan_switch_condition:
                    self.scan_switch = True
                    self.scan_switch_condition.notify_all()
        except asyncio.CancelledError:
            pass
    
    async def scan_file(self):
        try:
            while True:
                async with self.scan_switch_condition:
                    await self.scan_switch_condition.wait_for(lambda: self.scan_switch)
                    self.scan_switch = False
                    async with self.file_io_lock:
                        async with self.task_pool_lock:
                            await asyncio.shield(self.sync_file())
        except asyncio.CancelledError:
            await self.sync_file(should_stop=True)
    
    async def sync_file(self, should_stop=False):
        for file_path in self.jobs_dir.rglob("*.txt"):
            target_path = self.processed_dir / file_path.relative_to(self.jobs_dir).with_suffix('.old')
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            os_mtime = file_path.stat().st_mtime
            internal_mtime = self.file_last_mtime[file_path]
            complete = True
            first_line_flag = True
            for line in fileinput.input(str(file_path), inplace=True):
                if not line.strip() == "" and len(line.split())>1:
                    task, file_status = parse_task(line)
                    if should_stop:
                        if file_status == "running":
                            line = self.task_to_line(task, "waiting", first_line_flag)
                        else:
                            line = self.task_to_line(task, file_status, first_line_flag)
                    else:
                        pool_status = self.query_pool_status(task)
                        if internal_mtime > 0.0 and internal_mtime != os_mtime:
                            if file_status == "waiting" and pool_status != "waiting":
                                self.task_queue.put_nowait(task)
                            self.change_pool_status(task, file_status)
                        else:
                            if pool_status is None and file_status != "finished":
                                self.change_pool_status(task, "waiting")
                                pool_status = "waiting"
                                await self.task_queue.put(task)
                            
                            if pool_status == "waiting" and file_status != "waiting":
                                await self.task_queue.put(task)

                            if pool_status == "finished" and file_status != "finished":
                                self.file_with_changes.add(file_path)
                            
                            if pool_status != "finished":
                                complete = False
                            
                            line = self.task_to_line(task, pool_status, first_line_flag)
                    
                    sys.stdout.write(line)
                    _, current_status = parse_task(line)
                    complete = complete and current_status == "finished"
                    first_line_flag = False

            if complete and should_stop:
                with file_path.open("r") as src, target_path.open("a") as dst:
                    dst.write(src.read())
                file_path.unlink()
            elif complete:
                with file_path.open("r") as src, target_path.open("a") as dst:
                    dst.write(src.read() + "\n")
            self.file_last_mtime[file_path] = file_path.stat().st_mtime

class ProcessorWorker:
    def __init__(self, gpu_id, max_memory=40, max_util=80, grace_period=120):
        self.gpu_id = gpu_id
        self.status_lock = asyncio.Lock()
        self.status = "starting"
        self.max_memory = max_memory
        self.max_util = max_util
        self.grace_period = grace_period
        self.last_task_time = None
        self.task_start_times = dict()
        self.memory = deque(maxlen=60)
        self.util = deque(maxlen=60)
        self.memory.append(0)
        self.util.append(0)
        self.running_proc = dict() 

    async def start(self, task_queue, message_queue, task_pool, task_pool_lock):
        self.task_queue = task_queue
        self.message_queue = message_queue
        self.task_pool = task_pool
        self.task_pool_lock = task_pool_lock
        self.schedule_loop = asyncio.create_task(self.schedule())
        self.status = "idle"

    async def stop(self):
        self.schedule_loop.cancel()
        await asyncio.gather(self.schedule_loop, return_exceptions=True)

    async def generate_status_info(self):
        try:
            async with self.status_lock:
                columns, rows = os.get_terminal_size()
                info = f"GPU {self.gpu_id} - Mem: {max(self.memory):.2f}%, Util: {sum(self.util)/len(self.util):.2f}%, Status: {self.status}\n"
                for task, proc in self.running_proc.items():
                    timestr = time.strftime(' %H:%M:%S', time.gmtime(time.time() - self.task_start_times[task]))
                    taskstr = f"      - {task}"
                    if len(taskstr) > columns-len(timestr):
                        taskstr = taskstr[:columns-len(timestr)-3] + "..."
                    info += taskstr.ljust(columns-len(timestr)) + timestr + "\n"
                async with gpus_lock:
                    if self.gpu_id not in available_gpus and not self.running_proc:
                        return None
                return info
        except asyncio.CancelledError:
            pass

    async def schedule(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                async with self.status_lock:
                    # Get gpu status
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                    memory_pct = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / float(pynvml.nvmlDeviceGetMemoryInfo(handle).total) * 100.0
                    util_pct = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.memory.append(memory_pct)
                    self.util.append(util_pct)
                    max_memory_pct = max(self.memory)
                    avg_util_pct = sum(self.util)/len(self.util)
                    self.last_task_time = max(self.task_start_times.values()) if self.task_start_times else None
                    
                    # Get system memory usage
                    system_memory_pct = get_system_memory_usage()
                    
                    async with gpus_lock:
                        if self.gpu_id in available_gpus \
                            and max_memory_pct < self.max_memory \
                            and avg_util_pct < self.max_util \
                            and (self.last_task_time is None or time.time() - self.last_task_time > self.grace_period):
                            self.status = "idle"

                    async with gpus_lock:
                        if self.status == "idle" and self.gpu_id in available_gpus:
                            # Check system memory before starting new task
                            if system_memory_pct < 60.0:
                                try:
                                    task = self.task_queue.get_nowait()
                                    async with self.task_pool_lock:
                                        if task in self.task_pool["waiting"]:
                                            proc = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={self.gpu_id} {task}", shell=True, stdout=LOG_FILE, stderr=LOG_FILE)
                                            self.running_proc[task] = proc
                                            self.task_start_times[task] = time.time()
                                            self.status = "running"
                                            self.message_queue.put_nowait((task, "running"))

                                except asyncio.QueueEmpty:
                                    pass

                    tasks_to_remove = []
                    # Get system memory usage for checking running tasks
                    system_memory_pct = get_system_memory_usage()
                    async with gpus_lock:
                        for task, proc in self.running_proc.items():
                            proc_status = proc.poll()
                            if proc_status is not None:
                                tasks_to_remove.append(task)
                                if proc_status == 0:
                                    status = "finished"
                                else:
                                    status = "terminated"
                                self.message_queue.put_nowait((task, status))
                                self.task_start_times.pop(task)
                            elif self.gpu_id not in available_gpus:
                                tasks_to_remove.append(task)
                                self.message_queue.put_nowait((task, "waiting"))
                                self.task_start_times.pop(task)
                                proc.terminate()
                            elif system_memory_pct >= 90.0:
                                # Stop task if system memory exceeds 90%
                                tasks_to_remove.append(task)
                                self.message_queue.put_nowait((task, "waiting"))
                                self.task_start_times.pop(task)
                                proc.terminate()
                            else:
                                async with self.task_pool_lock:
                                    if task not in self.task_pool["running"] and task not in self.task_pool["waiting"]:
                                        tasks_to_remove.append(task)
                                        self.task_start_times.pop(task)
                                        proc.terminate()

                    for task in tasks_to_remove:
                        self.running_proc.pop(task)

        except asyncio.CancelledError:
            for _, proc in self.running_proc.items():
                proc.terminate()

async def monitor_gpus():
    global available_gpus
    gpu_file = pathlib.Path("available_gpus.txt")
    if not gpu_file.exists():
        gpu_file.touch()
    while True:
        await asyncio.sleep(0.1)
        async with gpus_lock:
            if gpu_file.exists():
                with gpu_file.open("r") as f:
                    file_gpus = f.read().split()
                    if all(gpu.isdigit() for gpu in file_gpus):
                        available_gpus = [int(gpu) for gpu in file_gpus]

async def console_printer(gpu_workers, queue):
    while True:
        refresh_grace_time = asyncio.create_task(asyncio.sleep(0.1))

        columns, rows = os.get_terminal_size()
        total_info = ""
        for gpu_worker in gpu_workers:
            info = await gpu_worker.generate_status_info()
            if info is not None:
                total_info += info
        
        left_tasks = queue.qsize()
        system_memory_pct = get_system_memory_usage()
        console_logger.info("\033[2J\033[H")
        console_logger.info("-" * columns)
        console_logger.info(f"Task in queue: {left_tasks}, System Memory: {system_memory_pct:.2f}%")
        console_logger.info("-" * columns)
        console_logger.info(total_info)
        console_logger.info("-" * columns)
        sys.stderr.flush()

        await refresh_grace_time

async def main():
    parser = argparse.ArgumentParser(description='Async GPU task scheduler')
    parser.add_argument('--max-memory', type=float, default=80.0,
                        help='Maximum GPU memory usage percentage (default: 80.0)')
    parser.add_argument('--max-util', type=float, default=80.0,
                        help='Maximum GPU utilization percentage (default: 80.0)')
    parser.add_argument('--grace-period', type=int, default=180,
                        help='Grace period in seconds before starting new task after last task (default: 180)')
    args = parser.parse_args()
    
    monitor_gpu_loop = None
    task_io = None
    task_io_loop = None
    gpu_workers = None
    gpu_worker_loops = None
    console_printer_loop = None
    
    try:
        monitor_gpu_loop = asyncio.create_task(monitor_gpus())
        task_io = TaskIO()
        task_io_loop = asyncio.create_task(task_io.start())
        gpu_workers = [ProcessorWorker(gpu_id, args.max_memory, args.max_util, args.grace_period) for gpu_id in range(pynvml.nvmlDeviceGetCount())]
        gpu_worker_loops = [asyncio.create_task(gpu_worker.start(task_io.task_queue, task_io.message_queue, task_io.task_pool, task_io.task_pool_lock)) for gpu_worker in gpu_workers]
        console_printer_loop = asyncio.create_task(console_printer(gpu_workers, task_io.task_queue))
        await asyncio.sleep(float('inf'))
        
    except KeyboardInterrupt:
        pass

    finally:
        tasks_to_gather = []
        
        # Cancel console printer and monitor loops
        if console_printer_loop is not None:
            console_printer_loop.cancel()
            tasks_to_gather.append(console_printer_loop)
        if monitor_gpu_loop is not None:
            monitor_gpu_loop.cancel()
            tasks_to_gather.append(monitor_gpu_loop)
        
        # Cancel TaskIO internal loops
        if task_io is not None:
            if hasattr(task_io, 'scan_file_loop'):
                task_io.scan_file_loop.cancel()
            if hasattr(task_io, 'monitor_file_loop'):
                task_io.monitor_file_loop.cancel()
            if hasattr(task_io, 'task_manage_loop'):
                task_io.task_manage_loop.cancel()
            tasks_to_gather.append(task_io.stop())
        
        # Stop GPU workers (this will cancel their internal schedule_loop)
        gpu_stops = []
        if gpu_workers is not None:
            gpu_stops = [asyncio.create_task(gpu_worker.stop()) for gpu_worker in gpu_workers]
            tasks_to_gather.extend(gpu_stops)
        
        # Cancel GPU worker start loops
        if gpu_worker_loops is not None:
            for gpu_worker_loop in gpu_worker_loops:
                gpu_worker_loop.cancel()
            tasks_to_gather.extend(gpu_worker_loops)
        
        # Cancel TaskIO start loop
        if task_io_loop is not None:
            task_io_loop.cancel()
            tasks_to_gather.append(task_io_loop)
        
        # Wait for all tasks to finish
        if tasks_to_gather:
            await asyncio.gather(*tasks_to_gather, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass