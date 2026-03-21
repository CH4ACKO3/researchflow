"""
scripts/async_train.py
Asynchronous GPU job scheduler that consumes job lists and dispatches shell tasks to available GPUs.

## Workflow Position
This file implements the scheduling stage in the project workflow (job list execution). It watches
`jobs/`, updates task states, dispatches runnable tasks onto GPUs, writes runtime history logs to
`logs/history/`, and archives completed job files into `processed/`.

## Main Components
- `get_system_memory_usage`: Reads `/proc/meminfo` and returns system memory usage percentage.
- `parse_task`: Parses one task line from job files into `(command, status)`.
- `TaskIO`: Synchronizes task states between in-memory pools/queues and `jobs/*.txt` files.
- `ProcessorWorker`: Per-GPU scheduler/runner that monitors NVML metrics and starts/stops subprocesses.
- `monitor_gpus`: Watches `available_gpus.txt` and refreshes the global GPU allow-list.
- `console_printer`: Renders live terminal status for queue and worker states.
- `main`: Bootstraps all async loops and performs coordinated shutdown.

## Inputs
- CLI args: `--max-memory`, `--max-util`, `--grace-period`, `--max-retries`, `--debug`.
- Files:
  - `available_gpus.txt` for externally controlled GPU allow-list.
  - `jobs/**/*.txt` where each line is a shell task with optional status prefix (`!`, `#`, `?`).

## Outputs / Side Effects
- Updates `jobs/**/*.txt` in-place with normalized status markers.
- Moves fully finished job files to `processed/**/*.old`.
- Appends scheduler/task runtime logs to `logs/history/history.log`.
- Spawns and terminates subprocesses with GPU/CPU affinity environment settings.

## Key Dependencies
- `pynvml`: GPU memory/utilization polling.
- `asyncio`: Concurrent queueing, polling, process management, and lifecycle control.
- `logging.handlers.TimedRotatingFileHandler`: Daily rotating scheduler history logs.

## Notes
- Task states are mirrored between files and in-memory pools; file mtimes are used to detect
  external edits and choose reconciliation direction.
- `available_gpus.txt` acts as an external kill switch: when a GPU is removed, running tasks on it
  are terminated and pushed back to waiting.
"""

import asyncio
import sys
import os
import time
import datetime
import subprocess
import pynvml
import logging
import logging.handlers
import argparse
import threading
from collections import defaultdict, deque
import pathlib
import unicodedata

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
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
log_formatter = logging.Formatter(LOG_FORMAT)

log_file_path = pathlib.Path("logs/history/history.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)


# Unicode box-drawing: ╭─╮ ╰─╯ │ ├─┤ (rounded corners, T-junction for divider)
_BOX_TL, _BOX_H, _BOX_TR = "\u256d", "\u2500", "\u256e"
_BOX_L, _BOX_R = "\u2502", "\u2502"
_BOX_BL, _BOX_BR = "\u2570", "\u256f"
_BOX_DIV_L, _BOX_DIV_R = "\u251c", "\u2524"


class HistoryLogBuffer:
    """
    Buffered history log with scheduler / GPU / process blocks.
    Supports periodic rewrite and midnight rolling (only clears finished process blocks).
    """
    WIDTH = 160

    def __init__(self):
        self._lock = threading.Lock()
        self.scheduler_logs = []  # list of (asctime, levelname, message)
        self.gpus = defaultdict(lambda: {"meta": [], "processes": {}})
        self._last_roll_date = datetime.date.today()

    def append_scheduler(self, record):
        with self._lock:
            self.scheduler_logs.append((
                log_formatter.formatTime(record),
                record.levelname,
                record.getMessage(),
            ))

    def append_process(self, gpu_id, task, record):
        with self._lock:
            if task not in self.gpus[gpu_id]["processes"]:
                self.gpus[gpu_id]["processes"][task] = {"meta": [], "logs": [], "finished": False}
            self.gpus[gpu_id]["processes"][task]["logs"].append((
                log_formatter.formatTime(record),
                record.levelname,
                record.getMessage(),
            ))

    def mark_process_finished(self, gpu_id, task):
        with self._lock:
            if task in self.gpus[gpu_id]["processes"]:
                self.gpus[gpu_id]["processes"][task]["finished"] = True

    def update_process_meta(self, gpu_id, task, meta_lines):
        with self._lock:
            if task in self.gpus[gpu_id]["processes"]:
                self.gpus[gpu_id]["processes"][task]["meta"] = list(meta_lines)

    def _fold_line(self, text, width):
        """Fold long lines; width = content width (excluding | |)."""
        lines = []
        while text:
            if len(text) <= width:
                lines.append(text.ljust(width))
                break
            cut = text[:width].rfind(" ")
            cut = width if cut <= 0 else cut
            lines.append(text[:cut].ljust(width))
            text = "  " + text[cut:].lstrip()
        return lines

    def _box_top(self, title=None):
        h_len = self.WIDTH - 2
        if title:
            pad = h_len - 4 - len(title)
            return f"{_BOX_TL}{_BOX_H}{_BOX_H} {title} {_BOX_H * max(0, pad)}{_BOX_TR}\n"
        return f"{_BOX_TL}{_BOX_H * h_len}{_BOX_TR}\n"

    def _box_bottom(self):
        return f"{_BOX_BL}{_BOX_H * (self.WIDTH - 2)}{_BOX_BR}\n"

    def _content_width(self):
        return self.WIDTH - 4

    def _wrap_content(self, meta_lines, log_lines):
        w = self._content_width()
        div_len = self.WIDTH - 2
        out = []
        for line in meta_lines:
            for folded in self._fold_line(line, w):
                out.append(f"{_BOX_L} {folded} {_BOX_R}\n")
        if meta_lines and log_lines:
            out.append(f"{_BOX_DIV_L}{_BOX_H * div_len}{_BOX_DIV_R}\n")
        for asctime, levelname, msg in log_lines:
            full = f"{asctime} - {levelname} - {msg}"
            for folded in self._fold_line(full, w):
                out.append(f"{_BOX_L} {folded} {_BOX_R}\n")
        return "".join(out)

    def _render_block(self, title, meta_lines, log_lines):
        out = self._box_top(title)
        out += self._wrap_content(meta_lines, log_lines)
        out += self._box_bottom()
        return out

    def _render_process_block_nested(self, meta_lines, log_lines, outer_w):
        """Render process block nested inside GPU block. outer_w = parent content width."""
        prefix = " "
        inner_w = outer_w - 5 - len(prefix)  # content: prefix + │ + sp + folded + sp + │
        h_len = inner_w + 2  # horizontal line spans (sp + content + sp) inside box
        lines = []
        lines.append(f"{prefix}{_BOX_TL}{_BOX_H * h_len}{_BOX_TR}")
        for line in meta_lines:
            for folded in self._fold_line(line, inner_w):
                lines.append(f"{prefix}{_BOX_L} {folded} {_BOX_R}")
        if meta_lines and log_lines:
            lines.append(f"{prefix}{_BOX_DIV_L}{_BOX_H * h_len}{_BOX_DIV_R}")
        for asctime, levelname, msg in log_lines:
            full = f"{asctime} - {levelname} - {msg}"
            for folded in self._fold_line(full, inner_w):
                lines.append(f"{prefix}{_BOX_L} {folded} {_BOX_R}")
        lines.append(f"{prefix}{_BOX_BL}{_BOX_H * h_len}{_BOX_BR}")
        return lines

    def _render_gpu_block_with_processes(self, gpu_id, gpu_meta, processes_data):
        """Render GPU block with process blocks nested inside."""
        w = self._content_width()
        out = self._box_top(f"GPU {gpu_id}")
        for line in gpu_meta:
            for folded in self._fold_line(line, w):
                out += f"{_BOX_L} {folded} {_BOX_R}\n"
        for task, proc_data in sorted(processes_data.items(), key=lambda x: x[0]):
            proc_meta = proc_data["meta"]
            proc_logs = proc_data["logs"]
            nested_lines = self._render_process_block_nested(proc_meta, proc_logs, w)
            for ln in nested_lines:
                out += f"{_BOX_L} {ln.ljust(w)} {_BOX_R}\n"
        out += self._box_bottom()
        return out

    def render(self, scheduler_meta, gpu_meta_dict=None):
        with self._lock:
            out = self._render_block("Scheduler", scheduler_meta, self.scheduler_logs)
            gpu_meta_dict = gpu_meta_dict or {}
            all_gpu_ids = sorted(set(self.gpus.keys()) | set(gpu_meta_dict.keys()))
            for gpu_id in all_gpu_ids:
                gpu_data = self.gpus.get(gpu_id, {"processes": {}})
                gpu_meta = gpu_meta_dict.get(gpu_id, [])
                out += self._render_gpu_block_with_processes(
                    gpu_id, gpu_meta, gpu_data["processes"]
                )
            return out

    def clear_finished_processes(self):
        with self._lock:
            for gpu_id in list(self.gpus.keys()):
                procs = self.gpus[gpu_id]["processes"]
                finished = [t for t, d in procs.items() if d["finished"]]
                for t in finished:
                    del procs[t]
                if not procs and not self.gpus[gpu_id]["meta"]:
                    del self.gpus[gpu_id]

    def check_and_roll(self, scheduler_meta=None, gpu_meta_dict=None):
        """
        At midnight, archive current content and clear finished process blocks.
        Returns True if roll was performed.
        """
        today = datetime.date.today()
        if today <= self._last_roll_date:
            return False
        self._last_roll_date = today
        meta_s = scheduler_meta or []
        meta_g = gpu_meta_dict or {}
        content = self.render(meta_s, meta_g)
        if content.strip():
            archive_path = log_file_path.with_name(f"{log_file_path.name}.{today}")
            try:
                with archive_path.open("a", encoding="utf-8") as f:
                    if archive_path.stat().st_size > 0:
                        f.write("\n")
                    f.write(content)
            except Exception as e:
                history_logger.debug(f"Failed to archive history: {e}")
        self.clear_finished_processes()
        return True


class BufferedHistoryHandler(logging.Handler):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record):
        try:
            gpu_id = getattr(record, "gpu_id", None)
            task = getattr(record, "task", None)
            if gpu_id is not None and task is not None:
                self.buffer.append_process(gpu_id, task, record)
            else:
                self.buffer.append_scheduler(record)
        except Exception:
            self.handleError(record)


def migrate_latest_history_log(log_file_path):
    """
    On startup, move existing history.log content to the latest archived log.
    """
    try:
        if (not log_file_path.exists()) or log_file_path.stat().st_size == 0:
            return

        archive_candidates = sorted(
            log_file_path.parent.glob(f"{log_file_path.name}.*"),
            key=lambda p: p.stat().st_mtime,
        )
        if archive_candidates:
            archive_path = archive_candidates[-1]
        else:
            ts = datetime.datetime.now().strftime("%Y-%m-%d")
            archive_path = log_file_path.with_name(f"{log_file_path.name}.{ts}.startup")

        with log_file_path.open("r", encoding="utf-8", errors="replace") as src:
            old_content = src.read()
        if not old_content:
            return
        if not old_content.endswith("\n"):
            old_content += "\n"

        with archive_path.open("a+", encoding="utf-8") as dst:
            dst.seek(0, os.SEEK_END)
            size = dst.tell()
            if size > 0:
                dst.seek(size - 1)
                if dst.read(1) != "\n":
                    dst.write("\n")
            dst.write(old_content)

        with log_file_path.open("w", encoding="utf-8"):
            pass
    except Exception as e:
        print(f"Failed to migrate history.log on startup: {e}", file=sys.stderr)


history_buffer = HistoryLogBuffer()
migrate_latest_history_log(log_file_path)

history_logger = logging.getLogger()
history_logger.setLevel(logging.INFO)
history_logger.handlers.clear()
buffered_handler = BufferedHistoryHandler(history_buffer)
buffered_handler.setLevel(logging.INFO)
history_logger.addHandler(buffered_handler)

# Keep file_handler ref for debug level changes in main()
file_handler = None

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
exclude_gpus = set()  # GPUs to exclude (e.g. {0, 1}), applied when reading available_gpus.txt

def parse_task(line):
    # Prefix markers in job files: ! running, # finished, ? terminated, no prefix waiting.
    if line.startswith("!"):
        return line[1:].strip(), "running"
    elif line.startswith("#"):
        return line[1:].strip(), "finished"
    elif line.startswith("?"):
        return line[1:].strip(), "terminated"
    else:
        return line.strip(), "waiting"

def is_stage_barrier(line):
    # A single '&' line acts as a stage barrier.
    # Also tolerate accidental status prefixes ("# &", "! &", "? &") from old states.
    stripped = line.strip()
    if stripped == "&":
        return True
    if stripped and stripped[0] in "!#?":
        return stripped[1:].strip() == "&"
    return False

def text_display_width(text):
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1
    return width

def fit_text_to_width(text, max_width):
    if max_width <= 0:
        return ""
    if text_display_width(text) <= max_width:
        return text
    ellipsis = "..."
    ellipsis_width = text_display_width(ellipsis)
    if max_width <= ellipsis_width:
        return "." * max_width
    kept = []
    used = 0
    for ch in text:
        ch_width = 0 if unicodedata.combining(ch) else (2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1)
        if used + ch_width + ellipsis_width > max_width:
            break
        kept.append(ch)
        used += ch_width
    return "".join(kept) + ellipsis

def pad_text_display(text, target_width):
    visible = fit_text_to_width(text, target_width)
    pad = max(0, target_width - text_display_width(visible))
    return visible + (" " * pad)

class HybridTaskQueue:
    """
    Queue with a front lane for urgent requeue.
    - Normal tasks go to `back_queue` (FIFO).
    - Urgent tasks use `front_queue` and are consumed first.
    """
    def __init__(self):
        self.front_queue = deque()
        self.back_queue = asyncio.Queue()

    async def put(self, task):
        # Keep async signature to match asyncio.Queue usage in existing code.
        self.back_queue.put_nowait(task)

    def put_nowait(self, task):
        self.back_queue.put_nowait(task)

    async def put_front(self, task):
        self.front_queue.append(task)

    def put_front_nowait(self, task):
        self.front_queue.append(task)

    def get_nowait(self):
        if self.front_queue:
            return self.front_queue.popleft()
        return self.back_queue.get_nowait()

    def qsize(self):
        return len(self.front_queue) + self.back_queue.qsize()

class TaskIO:
    def __init__(self, jobs_dir="jobs", processed_dir="processed"):
        self.jobs_dir = pathlib.Path(jobs_dir)
        self.processed_dir = pathlib.Path(processed_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup async data structures
        self.task_queue = HybridTaskQueue()
        self.message_queue = asyncio.Queue()
        self.task_pool = defaultdict(set)
        self.task_pool_lock = asyncio.Lock()
        self.file_with_changes = set()
        self.file_last_mtime = defaultdict(float)  # Key: Path object
        self.file_blocked_waiting_tasks = defaultdict(set)  # Key: Path object -> blocked waiting task strings
        self.file_task_stats = {}
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

    def task_to_line(self, task, status):
        if status == "finished":
            return f"# {task}"
        elif status == "terminated":
            return f"? {task}"
        elif status == "running":
            return f"! {task}"
        else:
            return task

    @staticmethod
    def render_lines(lines):
        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    @staticmethod
    def write_lines_atomically(file_path, lines):
        content = TaskIO.render_lines(lines)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, file_path)

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
                prior_stages_finished = True
                current_stage_finished = True
                previously_blocked = self.file_blocked_waiting_tasks[file_path]
                current_blocked = set()
                output_lines = []
                
                try:
                    with file_path.open("r", encoding="utf-8") as src:
                        source_lines = src.readlines()
                    for line in source_lines:
                        try:
                            stripped = line.strip()
                            if stripped == "":
                                continue

                            if is_stage_barrier(line):
                                output_lines.append("&")
                                complete = complete and (prior_stages_finished and current_stage_finished)
                                prior_stages_finished = prior_stages_finished and current_stage_finished
                                current_stage_finished = True
                                continue

                            if stripped:
                                task, file_status = parse_task(line)
                                task_blocked = not prior_stages_finished
                                if should_stop:
                                    if file_status == "running":
                                        line = self.task_to_line(task, "waiting")
                                    else:
                                        line = self.task_to_line(task, file_status)
                                    _, current_status = parse_task(line)
                                else:
                                    pool_status = self.query_pool_status(task)
                                    if internal_mtime > 0.0 and internal_mtime != os_mtime:
                                        # Job file was externally modified; file status takes precedence.
                                        if file_status == "waiting" and pool_status != "waiting" and not task_blocked:
                                            self.task_queue.put_nowait(task)
                                        self.change_pool_status(task, file_status)
                                        current_status = file_status
                                    else:
                                        # Initial bootstrap or unchanged file; in-memory pool drives status.
                                        if pool_status is None:
                                            # New task discovered; initialize pool and queue state.
                                            if file_status == "finished":
                                                # Keep completed tasks out of the queue.
                                                self.change_pool_status(task, "finished")
                                                pool_status = "finished"
                                            elif file_status == "terminated":
                                                # Keep terminated tasks terminated; do not auto-retry.
                                                self.change_pool_status(task, "terminated")
                                                pool_status = "terminated"
                                            else:
                                                # Normalize waiting/running file states into queue-backed waiting.
                                                self.change_pool_status(task, "waiting")
                                                pool_status = "waiting"
                                                if not task_blocked:
                                                    await self.task_queue.put(task)

                                        if pool_status == "waiting" and (not task_blocked) and task in previously_blocked:
                                            # Task was blocked by a previous '&' stage; enqueue once after unlock.
                                            await self.task_queue.put(task)

                                        if pool_status == "finished" and file_status != "finished":
                                            self.file_with_changes.add(file_path)
                                        
                                        if pool_status != "finished":
                                            complete = False
                                        
                                        line = self.task_to_line(task, pool_status)
                                        current_status = pool_status
                                
                                if task_blocked and current_status == "waiting":
                                    current_blocked.add(task)

                                output_lines.append(line)
                                complete = complete and current_status == "finished"
                                current_stage_finished = current_stage_finished and current_status == "finished"
                        except Exception as e:
                            history_logger.error(f"Error processing line in {file_path}: {e}")
                            output_lines.append(line.rstrip("\n"))
                except Exception as e:
                    history_logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                    continue

                self.file_blocked_waiting_tasks[file_path] = current_blocked
                file_stats = {"waiting": 0, "running": 0, "finished": 0}
                for out_line in output_lines:
                    if out_line == "&":
                        continue
                    _, status = parse_task(out_line)
                    if status == "running":
                        file_stats["running"] += 1
                    elif status == "finished":
                        file_stats["finished"] += 1
                    else:
                        # Treat waiting and terminated as "remaining" from file perspective.
                        file_stats["waiting"] += 1
                self.file_task_stats[file_path] = file_stats

                try:
                    if complete:
                        content = self.render_lines(output_lines)
                        with target_path.open("a", encoding="utf-8") as dst:
                            dst.write(content)
                            if not should_stop:
                                dst.write("\n")
                        file_path.unlink()
                        self.file_blocked_waiting_tasks.pop(file_path, None)
                        self.file_task_stats.pop(file_path, None)
                    else:
                        self.write_lines_atomically(file_path, output_lines)
                        self.file_last_mtime[file_path] = file_path.stat().st_mtime
                except (OSError, IOError) as e:
                    history_logger.error(f"Error archiving file {file_path}: {e}")
            except Exception as e:
                history_logger.error(f"Error syncing file {file_path}: {e}", exc_info=True)

class ProcessorWorker:
    def __init__(self, gpu_id, max_memory=40, max_util=80, grace_period=120, max_retries=3, debug=False):
        self.gpu_id = gpu_id
        self.status_lock = asyncio.Lock()
        self.status = "starting"
        self.max_memory = max_memory
        self.max_util = max_util
        self.grace_period = grace_period
        self.max_retries = max_retries
        self.last_task_time = None
        self.last_task_end_time = None
        self.task_start_times = dict()
        self.memory = deque(maxlen=60)
        self.util = deque(maxlen=60)
        self.memory.append(0)
        self.util.append(0)
        self.running_proc = dict()
        self.log_readers = dict()  # task -> (stdout_task, stderr_task)
        self.retry_counts = defaultdict(int)  # task -> retry count for non-zero exits
        self.debug = debug
        self.last_occupied_time = None
        
        self.num_cores = os.cpu_count() // 8
        self.cores = range(self.gpu_id * self.num_cores, (self.gpu_id + 1) * self.num_cores)

    async def start(self, task_queue, message_queue, task_pool, task_pool_lock):
        self.task_queue = task_queue
        self.message_queue = message_queue
        self.task_pool = task_pool
        self.task_pool_lock = task_pool_lock
        self.schedule_loop = asyncio.create_task(self.schedule())
        # self.status = "idle"

    async def stop(self):
        self.schedule_loop.cancel()
        await asyncio.gather(self.schedule_loop, return_exceptions=True)

    async def read_stream(self, task, stream, stream_name):
        """Async read subprocess output and log it"""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='replace').rstrip()
                if line_str:
                    bracket_pos = line_str.find('[')
                    if bracket_pos >= 0:
                        line_str = line_str[bracket_pos:]
                    history_logger.info(
                        line_str,
                        extra={"gpu_id": self.gpu_id, "task": task},
                    )
        except Exception as e:
            history_logger.debug(
                f"Error reading {stream_name}: {e}",
                extra={"gpu_id": self.gpu_id, "task": task},
            )

    async def generate_status_info(self):
        try:
            async with self.status_lock:
                columns, rows = os.get_terminal_size()

                elapsed_occupied_time = time.time() - self.last_occupied_time if (self.status == "occupied" and self.last_occupied_time is not None) else None
                occupied_str = f", waited {elapsed_occupied_time:.0f}s" if elapsed_occupied_time is not None else ""

                info = f"GPU {self.gpu_id} - Mem: {max(self.memory) if self.memory else 0.0:.2f}%, Util: {sum(self.util)/len(self.util) if self.util else 0.0:.2f}%, Status: {self.status}{occupied_str}\n"
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
                        max_memory_pct = max(self.memory) if self.memory else 0.0
                        avg_util_pct = sum(self.util)/len(self.util) if self.util else 0.0
                        self.last_task_time = max(self.task_start_times.values()) if self.task_start_times else None
                    
                        system_memory_pct = get_system_memory_usage()
                        
                        async with gpus_lock:
                            self_available = self.gpu_id in available_gpus
                        
                        # Detect external occupancy: no managed process but sustained usage not caused by recent completion.
                        time_since_last_end = (time.time() - self.last_task_end_time) if self.last_task_end_time else float('inf')
                        if (self.last_task_time is None) and (not self.running_proc) and (avg_util_pct > 10 or max_memory_pct > 10) and time_since_last_end > 30:
                            if self.status != "occupied":  # Record entry time only once per occupied period.
                                self.status = "occupied"
                                self.last_occupied_time = time.time()
                        elif self_available \
                            and max_memory_pct < self.max_memory \
                            and avg_util_pct < self.max_util \
                            and system_memory_pct < 80.0 \
                            and (self.last_task_time is None or time.time() - self.last_task_time > self.grace_period) \
                            and (self.last_occupied_time is None or time.time() - self.last_occupied_time > 15):
                            self.status = "idle"
                            
                        if self.status == "idle":
                            try:
                                task = self.task_queue.get_nowait()
                                async with self.task_pool_lock:
                                    if task in self.task_pool["waiting"]:
                                        try:
                                            env = os.environ.copy()
                                            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
                                            env["OMP_NUM_THREADS"] = "4"
                                            env["MKL_NUM_THREADS"] = "4"
                                            
                                            proc = await asyncio.create_subprocess_shell(
                                                f"taskset -c {self.cores[0]}-{self.cores[-1]} {task}", 
                                                stdout=asyncio.subprocess.PIPE, 
                                                stderr=asyncio.subprocess.PIPE,
                                                env=env
                                            )
                                            self.running_proc[task] = proc
                                            self.task_start_times[task] = time.time()
                                            
                                            # Start async tasks to read stdout and stderr
                                            stdout_task = asyncio.create_task(self.read_stream(task, proc.stdout, "stdout"))
                                            stderr_task = asyncio.create_task(self.read_stream(task, proc.stderr, "stderr"))
                                            self.log_readers[task] = (stdout_task, stderr_task)
                                            
                                            self.status = "running"
                                            self.message_queue.put_nowait((task, "running"))
                                            history_logger.info(
                                                f"started task: {task}",
                                                extra={"gpu_id": self.gpu_id, "task": task},
                                            )
                                        except (OSError, subprocess.SubprocessError) as e:
                                            history_logger.error(
                                                f"Failed to start task: {e}",
                                                extra={"gpu_id": self.gpu_id, "task": task},
                                            )
                                            self.message_queue.put_nowait((task, "waiting"))
                            except asyncio.QueueEmpty:
                                pass
                            
                        tasks_to_remove = []
                        running_proc_copy = dict(self.running_proc)

                        for task, proc in running_proc_copy.items():
                            try:
                                proc_status = proc.returncode
                                if proc_status is not None:
                                    tasks_to_remove.append(task)
                                    if proc_status == 0:
                                        self.retry_counts.pop(task, None)
                                        status = "finished"
                                        history_buffer.mark_process_finished(self.gpu_id, task)
                                        history_logger.info(
                                            f"finished task: {task}",
                                            extra={"gpu_id": self.gpu_id, "task": task},
                                        )
                                    else:
                                        retry_count = self.retry_counts[task] + 1
                                        if retry_count <= self.max_retries:
                                            self.retry_counts[task] = retry_count
                                            # Generic subprocess failure: retry from queue tail.
                                            async with self.task_pool_lock:
                                                self.task_pool["running"].discard(task)
                                                self.task_pool["waiting"].add(task)
                                            self.task_queue.put_nowait(task)
                                            status = "waiting"
                                            history_logger.warning(
                                                f"task failed with code {proc_status}, retry "
                                                f"{retry_count}/{self.max_retries}: {task}",
                                                extra={"gpu_id": self.gpu_id, "task": task},
                                            )
                                        else:
                                            self.retry_counts.pop(task, None)
                                            status = "terminated"
                                            history_buffer.mark_process_finished(self.gpu_id, task)
                                            history_logger.warning(
                                                f"task failed with code {proc_status}, "
                                                f"retries exhausted ({self.max_retries}): {task}",
                                                extra={"gpu_id": self.gpu_id, "task": task},
                                            )
                                    self.message_queue.put_nowait((task, status))
                                    self.task_start_times.pop(task, None)
                                elif not self_available:
                                    tasks_to_remove.append(task)
                                    # GPU removed externally: interrupt and urgently requeue at queue front.
                                    async with self.task_pool_lock:
                                        if task in self.task_pool["running"]:
                                            self.task_pool["running"].discard(task)
                                            self.task_pool["waiting"].add(task)
                                    self.task_queue.put_front_nowait(task)
                                    self.message_queue.put_nowait((task, "waiting"))
                                    self.task_start_times.pop(task, None)
                                    proc.terminate()
                                    history_buffer.mark_process_finished(self.gpu_id, task)
                                    history_logger.info(
                                        f"terminated task (GPU unavailable): {task}",
                                        extra={"gpu_id": self.gpu_id, "task": task},
                                    )
                                elif system_memory_pct >= 90.0:
                                    # Stop task if system memory exceeds 90%
                                    tasks_to_remove.append(task)
                                    self.message_queue.put_nowait((task, "waiting"))
                                    self.task_start_times.pop(task, None)
                                    proc.terminate()
                                    history_buffer.mark_process_finished(self.gpu_id, task)
                                    history_logger.warning(
                                        f"terminated task (high system memory): {task}",
                                        extra={"gpu_id": self.gpu_id, "task": task},
                                    )
                                else:
                                    async with self.task_pool_lock:
                                        if task not in self.task_pool["running"] and task not in self.task_pool["waiting"]:
                                            tasks_to_remove.append(task)
                                            self.task_start_times.pop(task, None)
                                            proc.terminate()
                                            history_buffer.mark_process_finished(self.gpu_id, task)
                                            history_logger.info(
                                                f"terminated task (removed from pool): {task}",
                                                extra={"gpu_id": self.gpu_id, "task": task},
                                            )
                            except Exception as e:
                                history_logger.error(
                                    f"Error checking task status: {e}",
                                    extra={"gpu_id": self.gpu_id, "task": task},
                                )
                            
                        for task in tasks_to_remove:
                            self.running_proc.pop(task, None)
                            # Cancel and cleanup log reader tasks
                            if task in self.log_readers:
                                stdout_task, stderr_task = self.log_readers.pop(task)
                                stdout_task.cancel()
                                stderr_task.cancel()
                        
                        if tasks_to_remove:
                            self.last_task_end_time = time.time()
                            
                        if not self.running_proc:
                            self.memory.clear()
                            self.util.clear()

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
                        history_buffer.mark_process_finished(self.gpu_id, task)
                        history_logger.info(
                            f"terminated task during shutdown: {task}",
                            extra={"gpu_id": self.gpu_id, "task": task},
                        )
                    except Exception as e:
                        history_logger.error(
                            f"Error terminating process: {e}",
                            extra={"gpu_id": self.gpu_id, "task": task},
                        )
                
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
                                    raw = [int(gpu) for gpu in file_gpus]
                                    available_gpus = [g for g in raw if g not in exclude_gpus]
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

async def console_printer(gpu_workers, task_io):
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
                
                system_memory_pct = get_system_memory_usage()
                
                try:
                    console_logger.info("\033[2J\033[H")
                    console_logger.info("-" * columns)
                    console_logger.info(f"System Memory: {system_memory_pct:.2f}%")
                    async with task_io.file_io_lock:
                        job_stats = sorted(task_io.file_task_stats.items(), key=lambda x: str(x[0]))
                    if not job_stats:
                        console_logger.info("No active job files")
                    else:
                        rel_names = [str(file_path.relative_to(task_io.jobs_dir)) for file_path, _ in job_stats]
                        max_name_len = max(text_display_width("TOTAL"), *(text_display_width(name) for name in rel_names))
                        fixed_metric_width = len(" | waiting: 00000 | running: 00000 | finished: 00000")
                        max_allowed_name = max(12, columns - fixed_metric_width)
                        name_width = min(max_name_len, max_allowed_name)

                        total_waiting = 0
                        total_running = 0
                        total_finished = 0

                        def format_row(name, waiting, running, finished):
                            name_col = pad_text_display(name, name_width)
                            return (
                                f"{name_col} | waiting: {waiting:5d} | "
                                f"running: {running:5d} | finished: {finished:5d}"
                            )

                        for file_path, stats in job_stats:
                            relative_name = str(file_path.relative_to(task_io.jobs_dir))
                            waiting = stats["waiting"]
                            running = stats["running"]
                            finished = stats["finished"]
                            total_waiting += waiting
                            total_running += running
                            total_finished += finished
                            console_logger.info(format_row(relative_name, waiting, running, finished))

                        console_logger.info(
                            format_row("TOTAL", total_waiting, total_running, total_finished)
                        )
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


HISTORY_FLUSH_INTERVAL = 2.0


async def _gather_scheduler_meta(task_io):
    lines = []
    system_memory_pct = get_system_memory_usage()
    lines.append(f"System Memory: {system_memory_pct:.2f}%")
    async with task_io.file_io_lock:
        job_stats = sorted(task_io.file_task_stats.items(), key=lambda x: str(x[0]))
    if not job_stats:
        lines.append("No active job files")
    else:
        for file_path, stats in job_stats:
            rel = str(file_path.relative_to(task_io.jobs_dir))
            lines.append(
                f"{rel} | waiting: {stats['waiting']:5d} | "
                f"running: {stats['running']:5d} | finished: {stats['finished']:5d}"
            )
        total = {"waiting": 0, "running": 0, "finished": 0}
        for _, s in job_stats:
            total["waiting"] += s["waiting"]
            total["running"] += s["running"]
            total["finished"] += s["finished"]
        lines.append(
            f"TOTAL | waiting: {total['waiting']:5d} | "
            f"running: {total['running']:5d} | finished: {total['finished']:5d}"
        )
    return lines


async def _gather_gpu_meta(gpu_workers):
    result = {}
    for w in gpu_workers:
        try:
            info = await w.generate_status_info()
            if info is not None:
                result[w.gpu_id] = [ln.strip() for ln in info.strip().split("\n") if ln.strip()]
        except Exception:
            result[w.gpu_id] = [f"GPU {w.gpu_id} - Status: {w.status}"]
    return result


async def history_flush_loop(task_io, gpu_workers):
    _last_content = None
    try:
        while True:
            await asyncio.sleep(HISTORY_FLUSH_INTERVAL)
            try:
                scheduler_meta = await _gather_scheduler_meta(task_io)
                gpu_meta = await _gather_gpu_meta(gpu_workers)

                history_buffer.check_and_roll(scheduler_meta, gpu_meta)

                for w in gpu_workers:
                    async with w.status_lock:
                        for task in w.running_proc:
                            elapsed = time.time() - w.task_start_times.get(task, time.time())
                            timestr = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                            meta = [f"Command: {task}", f"Runtime: {timestr}"]
                            history_buffer.update_process_meta(w.gpu_id, task, meta)

                content = history_buffer.render(scheduler_meta, gpu_meta)
                if content != _last_content:
                    with log_file_path.open("w", encoding="utf-8") as f:
                        f.write(content)
                    _last_content = content
            except Exception as e:
                history_logger.debug(f"Error in history flush: {e}")
    except asyncio.CancelledError:
        history_logger.debug("history_flush_loop cancelled")
    finally:
        try:
            scheduler_meta = await _gather_scheduler_meta(task_io)
            gpu_meta = await _gather_gpu_meta(gpu_workers) if gpu_workers else {}
            content = history_buffer.render(scheduler_meta, gpu_meta)
            with log_file_path.open("w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            history_logger.debug(f"Final history flush error: {e}")


async def main():
    parser = argparse.ArgumentParser(description='Async GPU task scheduler')
    parser.add_argument('--max-memory', type=float, default=80.0,
                        help='Maximum GPU memory usage percentage (default: 80.0)')
    parser.add_argument('--max-util', type=float, default=80.0,
                        help='Maximum GPU utilization percentage (default: 80.0)')
    parser.add_argument('--grace-period', type=int, default=180,
                        help='Grace period in seconds before starting new task after last task (default: 180)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum retries for non-zero exit tasks (default: 3)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode')
    parser.add_argument('--exclude-gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to exclude (e.g. 0,1)')
    args = parser.parse_args()

    if args.exclude_gpus:
        exclude_gpus.update(int(x.strip()) for x in args.exclude_gpus.split(",") if x.strip())
    
    if args.debug:
        console_logger.setLevel(logging.DEBUG)
        buffered_handler.setLevel(logging.DEBUG)
        history_logger.setLevel(logging.DEBUG)
    
    history_logger.info("Starting async GPU task scheduler")
    
    monitor_gpu_loop = None
    task_io = None
    task_io_loop = None
    gpu_workers = None
    gpu_worker_loops = None
    console_printer_loop = None
    history_flush_loop_task = None

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
        
        gpu_workers = [
            ProcessorWorker(gpu_id, args.max_memory, args.max_util, args.grace_period, args.max_retries, args.debug)
            for gpu_id in range(gpu_count)
        ]
        gpu_worker_loops = [asyncio.create_task(gpu_worker.start(task_io.task_queue, task_io.message_queue, task_io.task_pool, task_io.task_pool_lock)) for gpu_worker in gpu_workers]
        console_printer_loop = asyncio.create_task(console_printer(gpu_workers, task_io))
        history_flush_loop_task = asyncio.create_task(history_flush_loop(task_io, gpu_workers))
        
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
        if history_flush_loop_task is not None:
            history_logger.debug("Cancelling history flush loop")
            history_flush_loop_task.cancel()
            tasks_to_gather.append(history_flush_loop_task)
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