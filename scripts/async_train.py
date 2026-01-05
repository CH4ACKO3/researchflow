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
                try:
                    grace_time = asyncio.create_task(asyncio.sleep(1))
                    any_change = False
                    async with self.file_io_lock:
                        for file_path in self.jobs_dir.rglob("*.txt"):
                            try:
                                if file_path.stat().st_mtime > self.file_last_mtime[file_path]:
                                    any_change = True
                            except (OSError, IOError) as e:
                                history_logger.warning(f"Failed to check file {file_path}: {e}")
                                    
                    if any_change:
                        async with self.scan_switch_condition:
                            self.scan_switch = True
                            self.scan_switch_condition.notify_all()
                    
                    await grace_time
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    history_logger.error(f"Error in monitor_file loop: {e}", exc_info=True)
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            history_logger.info("monitor_file task cancelled")
        finally:
            history_logger.debug("monitor_file task stopped")

    async def task_manage(self):
        try:
            while True:
                try:
                    message = await self.message_queue.get()
                    async with self.task_pool_lock:
                        task, status = message
                        self.change_pool_status(task, status)
                    
                    async with self.scan_switch_condition:
                        self.scan_switch = True
                        self.scan_switch_condition.notify_all()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    history_logger.error(f"Error in task_manage loop: {e}", exc_info=True)
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            history_logger.info("task_manage task cancelled")
        finally:
            history_logger.debug("task_manage task stopped")
    
    async def scan_file(self):
        try:
            while True:
                try:
                    async with self.scan_switch_condition:
                        await self.scan_switch_condition.wait_for(lambda: self.scan_switch)
                        self.scan_switch = False
                        async with self.file_io_lock:
                            async with self.task_pool_lock:
                                await asyncio.shield(self.sync_file())
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    history_logger.error(f"Error in scan_file loop: {e}", exc_info=True)
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            history_logger.info("scan_file task cancelled, syncing files before stop")
            try:
                await self.sync_file(should_stop=True)
            except Exception as e:
                history_logger.error(f"Error during final sync_file: {e}", exc_info=True)
        finally:
            history_logger.debug("scan_file task stopped")
    
    async def sync_file(self, should_stop=False):
        for file_path in self.jobs_dir.rglob("*.txt"):
            try:
                target_path = self.processed_dir / file_path.relative_to(self.jobs_dir).with_suffix('.old')
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                os_mtime = file_path.stat().st_mtime
                internal_mtime = self.file_last_mtime[file_path]
                complete = True
                first_line_flag = True
                
                try:
                    for line in fileinput.input(str(file_path), inplace=True):
                        try:
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
                        except Exception as e:
                            history_logger.error(f"Error processing line in {file_path}: {e}")
                            sys.stdout.write(line)
                except Exception as e:
                    history_logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                    continue

                try:
                    if complete and should_stop:
                        with file_path.open("r") as src, target_path.open("a") as dst:
                            dst.write(src.read())
                        file_path.unlink()
                    elif complete:
                        with file_path.open("r") as src, target_path.open("a") as dst:
                            dst.write(src.read() + "\n")
                    self.file_last_mtime[file_path] = file_path.stat().st_mtime
                except (OSError, IOError) as e:
                    history_logger.error(f"Error archiving file {file_path}: {e}")
            except Exception as e:
                history_logger.error(f"Error syncing file {file_path}: {e}", exc_info=True)

class ProcessorWorker:
    def __init__(self, gpu_id, max_memory=40, max_util=80, grace_period=120, debug=False):
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
        self.log_readers = dict()  # task -> (stdout_task, stderr_task)
        self.debug = debug

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

    async def read_stream(self, task, stream, stream_name):
        """Async read subprocess output and log it"""
        try:
            loop = asyncio.get_event_loop()
            while True:
                line = await loop.run_in_executor(None, stream.readline)
                if not line:
                    break
                line_str = line.decode('utf-8', errors='replace').rstrip()
                if line_str:
                    history_logger.info(f"[GPU {self.gpu_id}] [{task}] {line_str}")
        except Exception as e:
            history_logger.debug(f"Error reading {stream_name} for task {task}: {e}")
        finally:
            try:
                stream.close()
            except:
                pass

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
        except OSError as e:
            history_logger.warning(f"Failed to get terminal size for GPU {self.gpu_id}: {e}")
            return f"GPU {self.gpu_id} - Status: {self.status}\n"
        except Exception as e:
            history_logger.error(f"Error generating status info for GPU {self.gpu_id}: {e}")
            return None

    async def schedule(self):
        try:
            while True:
                try:
                    await asyncio.sleep(0.1)
                    
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                        memory_pct = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / float(pynvml.nvmlDeviceGetMemoryInfo(handle).total) * 100.0
                        util_pct = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    except pynvml.NVMLError as e:
                        history_logger.error(f"NVML error for GPU {self.gpu_id}: {e}")
                        await asyncio.sleep(1)
                        continue
                    
                    async with self.status_lock:
                        self.memory.append(memory_pct)
                        self.util.append(util_pct)
                        max_memory_pct = max(self.memory)
                        avg_util_pct = sum(self.util)/len(self.util)
                        self.last_task_time = max(self.task_start_times.values()) if self.task_start_times else None
                    
                        system_memory_pct = get_system_memory_usage()
                        
                        async with gpus_lock:
                            self_available = self.gpu_id in available_gpus
                            
                        if self_available \
                            and max_memory_pct < self.max_memory \
                            and avg_util_pct < self.max_util \
                            and system_memory_pct < 60.0 \
                            and (self.last_task_time is None or time.time() - self.last_task_time > self.grace_period):
                            self.status = "idle"
                        elif self_available and self.last_task_time is None:
                            self.status = "occupied"
                            
                        if self.status == "idle":
                            try:
                                task = self.task_queue.get_nowait()
                                async with self.task_pool_lock:
                                    if task in self.task_pool["waiting"]:
                                        try:
                                            proc = subprocess.Popen(
                                                f"CUDA_VISIBLE_DEVICES={self.gpu_id} {task}{' --debug' if self.debug else ''}", 
                                                shell=True, 
                                                stdout=subprocess.PIPE, 
                                                stderr=subprocess.PIPE
                                            )
                                            self.running_proc[task] = proc
                                            self.task_start_times[task] = time.time()
                                            
                                            # Start async tasks to read stdout and stderr
                                            stdout_task = asyncio.create_task(self.read_stream(task, proc.stdout, "stdout"))
                                            stderr_task = asyncio.create_task(self.read_stream(task, proc.stderr, "stderr"))
                                            self.log_readers[task] = (stdout_task, stderr_task)
                                            
                                            self.status = "running"
                                            self.message_queue.put_nowait((task, "running"))
                                            history_logger.info(f"GPU {self.gpu_id} started task: {task}")
                                        except (OSError, subprocess.SubprocessError) as e:
                                            history_logger.error(f"Failed to start task on GPU {self.gpu_id}: {task}, error: {e}")
                                            self.message_queue.put_nowait((task, "waiting"))
                            except asyncio.QueueEmpty:
                                pass
                            
                        tasks_to_remove = []
                        running_proc_copy = dict(self.running_proc)

                        for task, proc in running_proc_copy.items():
                            try:
                                proc_status = proc.poll()
                                if proc_status is not None:
                                    tasks_to_remove.append(task)
                                    if proc_status == 0:
                                        status = "finished"
                                        history_logger.info(f"GPU {self.gpu_id} finished task: {task}")
                                    else:
                                        status = "terminated"
                                        history_logger.warning(f"GPU {self.gpu_id} terminated task with code {proc_status}: {task}")
                                    self.message_queue.put_nowait((task, status))
                                    self.task_start_times.pop(task, None)
                                elif not self_available:
                                    tasks_to_remove.append(task)
                                    self.message_queue.put_nowait((task, "waiting"))
                                    self.task_start_times.pop(task, None)
                                    proc.terminate()
                                    history_logger.info(f"GPU {self.gpu_id} terminated task (GPU unavailable): {task}")
                                elif system_memory_pct >= 90.0:
                                    # Stop task if system memory exceeds 90%
                                    tasks_to_remove.append(task)
                                    self.message_queue.put_nowait((task, "waiting"))
                                    self.task_start_times.pop(task, None)
                                    proc.terminate()
                                    history_logger.warning(f"GPU {self.gpu_id} terminated task (high system memory): {task}")
                                else:
                                    async with self.task_pool_lock:
                                        if task not in self.task_pool["running"] and task not in self.task_pool["waiting"]:
                                            tasks_to_remove.append(task)
                                            self.task_start_times.pop(task, None)
                                            proc.terminate()
                                            history_logger.info(f"GPU {self.gpu_id} terminated task (removed from pool): {task}")
                            except Exception as e:
                                history_logger.error(f"Error checking task status on GPU {self.gpu_id}: {task}, error: {e}")
                            
                            for task in tasks_to_remove:
                                self.running_proc.pop(task, None)
                                # Cancel and cleanup log reader tasks
                                if task in self.log_readers:
                                    stdout_task, stderr_task = self.log_readers.pop(task)
                                    stdout_task.cancel()
                                    stderr_task.cancel()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    history_logger.error(f"Error in schedule loop for GPU {self.gpu_id}: {e}", exc_info=True)
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            history_logger.info(f"GPU {self.gpu_id} schedule task cancelled")
        finally:
            history_logger.debug(f"GPU {self.gpu_id} terminating all running processes")
            async with self.status_lock:
                for task, proc in self.running_proc.items():
                    try:
                        proc.terminate()
                        history_logger.info(f"GPU {self.gpu_id} terminated task during shutdown: {task}")
                    except Exception as e:
                        history_logger.error(f"Error terminating process for task {task}: {e}")
                
                # Cancel all log reader tasks
                for task, (stdout_task, stderr_task) in self.log_readers.items():
                    stdout_task.cancel()
                    stderr_task.cancel()
                self.log_readers.clear()

async def monitor_gpus():
    global available_gpus
    gpu_file = pathlib.Path("available_gpus.txt")
    try:
        if not gpu_file.exists():
            gpu_file.touch()
    except (OSError, IOError) as e:
        history_logger.error(f"Failed to create GPU file: {e}")
    
    try:
        while True:
            try:
                await asyncio.sleep(0.1)
                async with gpus_lock:
                    try:
                        if gpu_file.exists():
                            with gpu_file.open("r") as f:
                                file_gpus = f.read().split()
                                if all(gpu.isdigit() for gpu in file_gpus):
                                    available_gpus = [int(gpu) for gpu in file_gpus]
                    except (OSError, IOError) as e:
                        history_logger.warning(f"Failed to read GPU file: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                history_logger.error(f"Error in monitor_gpus loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        history_logger.info("monitor_gpus task cancelled")
    finally:
        history_logger.debug("monitor_gpus task stopped")

async def console_printer(gpu_workers, queue):
    try:
        while True:
            try:
                refresh_grace_time = asyncio.create_task(asyncio.sleep(0.1))

                try:
                    columns, rows = os.get_terminal_size()
                except OSError:
                    columns = 80  # Default fallback
                    rows = 24
                
                total_info = ""
                for gpu_worker in gpu_workers:
                    try:
                        info = await gpu_worker.generate_status_info()
                        if info is not None:
                            total_info += info
                    except Exception as e:
                        history_logger.error(f"Error getting status info: {e}")
                
                left_tasks = queue.qsize()
                system_memory_pct = get_system_memory_usage()
                
                try:
                    console_logger.info("\033[2J\033[H")
                    console_logger.info("-" * columns)
                    console_logger.info(f"Task in queue: {left_tasks}, System Memory: {system_memory_pct:.2f}%")
                    console_logger.info("-" * columns)
                    console_logger.info(total_info)
                    console_logger.info("-" * columns)
                    sys.stderr.flush()
                except Exception as e:
                    history_logger.error(f"Error printing to console: {e}")

                await refresh_grace_time
            except asyncio.CancelledError:
                raise
            except Exception as e:
                history_logger.error(f"Error in console_printer loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        history_logger.info("console_printer task cancelled")
    finally:
        history_logger.debug("console_printer task stopped")

async def main():
    parser = argparse.ArgumentParser(description='Async GPU task scheduler')
    parser.add_argument('--max-memory', type=float, default=80.0,
                        help='Maximum GPU memory usage percentage (default: 80.0)')
    parser.add_argument('--max-util', type=float, default=80.0,
                        help='Maximum GPU utilization percentage (default: 80.0)')
    parser.add_argument('--grace-period', type=int, default=180,
                        help='Grace period in seconds before starting new task after last task (default: 180)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode')
    args = parser.parse_args()
    
    if args.debug:
        console_logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        history_logger.setLevel(logging.DEBUG)
    
    history_logger.info("Starting async GPU task scheduler")
    
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
        
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            history_logger.info(f"Detected {gpu_count} GPUs")
        except pynvml.NVMLError as e:
            history_logger.error(f"Failed to get GPU count: {e}")
            raise
        
        gpu_workers = [ProcessorWorker(gpu_id, args.max_memory, args.max_util, args.grace_period, args.debug) for gpu_id in range(gpu_count)]
        gpu_worker_loops = [asyncio.create_task(gpu_worker.start(task_io.task_queue, task_io.message_queue, task_io.task_pool, task_io.task_pool_lock)) for gpu_worker in gpu_workers]
        console_printer_loop = asyncio.create_task(console_printer(gpu_workers, task_io.task_queue))
        
        history_logger.info("All tasks started successfully")
        await asyncio.sleep(float('inf'))

    except asyncio.CancelledError:
        # This happens when asyncio.run() receives KeyboardInterrupt and cancels all tasks
        history_logger.info("Main task cancelled (keyboard interrupt)")
    except Exception as e:
        history_logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        raise
    finally:
        history_logger.info("Shutting down scheduler")
        tasks_to_gather = []
        
        # Cancel console printer and monitor loops
        if console_printer_loop is not None:
            history_logger.debug("Cancelling console printer")
            console_printer_loop.cancel()
            tasks_to_gather.append(console_printer_loop)
        if monitor_gpu_loop is not None:
            history_logger.debug("Cancelling GPU monitor")
            monitor_gpu_loop.cancel()
            tasks_to_gather.append(monitor_gpu_loop)
        
        # Cancel TaskIO internal loops
        if task_io is not None:
            history_logger.debug("Cancelling TaskIO loops")
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
            history_logger.debug("Stopping GPU workers")
            gpu_stops = [asyncio.create_task(gpu_worker.stop()) for gpu_worker in gpu_workers]
            tasks_to_gather.extend(gpu_stops)
        
        # Cancel GPU worker start loops
        if gpu_worker_loops is not None:
            history_logger.debug("Cancelling GPU worker loops")
            for gpu_worker_loop in gpu_worker_loops:
                gpu_worker_loop.cancel()
            tasks_to_gather.extend(gpu_worker_loops)
        
        # Cancel TaskIO start loop
        if task_io_loop is not None:
            history_logger.debug("Cancelling TaskIO start loop")
            task_io_loop.cancel()
            tasks_to_gather.append(task_io_loop)
        
        # Wait for all tasks to finish
        if tasks_to_gather:
            history_logger.debug(f"Waiting for {len(tasks_to_gather)} tasks to finish")
            results = await asyncio.gather(*tasks_to_gather, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    history_logger.error(f"Task {i} raised exception during cleanup: {result}")
        
        history_logger.info("Scheduler shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        history_logger.info("Program interrupted by user")
    except Exception as e:
        history_logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)