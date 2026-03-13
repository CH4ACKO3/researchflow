import asyncio
import importlib
import sys
import time
from collections import defaultdict
from types import SimpleNamespace


class _FakeProc:
    def __init__(self, returncode):
        self.returncode = returncode
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _load_async_train_module(monkeypatch, tmp_path):
    fake_pynvml = SimpleNamespace(
        NVMLError=RuntimeError,
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda idx: idx,
        nvmlDeviceGetMemoryInfo=lambda _handle: SimpleNamespace(used=0, total=100),
        nvmlDeviceGetUtilizationRates=lambda _handle: SimpleNamespace(gpu=0),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module("scripts.async_train")
    return importlib.reload(module)


def test_stage_barrier_recognizes_only_single_ampersand(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    assert m.is_stage_barrier("  &  \n")
    assert not m.is_stage_barrier("echo hi &\n")
    assert not m.is_stage_barrier("& echo hi\n")


def test_hybrid_task_queue_front_has_priority(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    q = m.HybridTaskQueue()
    q.put_nowait("normal_1")
    q.put_nowait("normal_2")
    q.put_front_nowait("urgent_1")
    q.put_front_nowait("urgent_2")

    assert q.get_nowait() == "urgent_1"
    assert q.get_nowait() == "urgent_2"
    assert q.get_nowait() == "normal_1"
    assert q.get_nowait() == "normal_2"


def test_sync_file_stage_barriers_support_multiple_sections(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    jobs_dir = tmp_path / "jobs"
    processed_dir = tmp_path / "processed"
    jobs_dir.mkdir()
    processed_dir.mkdir()

    job_file = jobs_dir / "stage.txt"
    job_file.write_text("task_a\n&\ntask_b\n&\ntask_c\n")

    io = m.TaskIO(jobs_dir=str(jobs_dir), processed_dir=str(processed_dir))
    asyncio.run(io.sync_file())
    assert io.task_queue.get_nowait() == "task_a"
    assert io.file_blocked_waiting_tasks[job_file] == {"task_b", "task_c"}

    io.change_pool_status("task_a", "finished")
    asyncio.run(io.sync_file())
    assert io.task_queue.get_nowait() == "task_b"
    assert io.file_blocked_waiting_tasks[job_file] == {"task_c"}

    io.change_pool_status("task_b", "finished")
    asyncio.run(io.sync_file())
    assert io.task_queue.get_nowait() == "task_c"
    assert io.file_blocked_waiting_tasks[job_file] == set()


def test_sync_file_keeps_terminated_tasks_terminated(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    jobs_dir = tmp_path / "jobs"
    processed_dir = tmp_path / "processed"
    jobs_dir.mkdir()
    processed_dir.mkdir()

    job_file = jobs_dir / "terminated.txt"
    job_file.write_text("? task_killed\n")

    io = m.TaskIO(jobs_dir=str(jobs_dir), processed_dir=str(processed_dir))
    asyncio.run(io.sync_file())

    assert "task_killed" in io.task_pool["terminated"]
    assert io.task_queue.qsize() == 0
    assert job_file.read_text().strip() == "? task_killed"


def test_sync_file_inline_ampersand_is_not_barrier(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    jobs_dir = tmp_path / "jobs"
    processed_dir = tmp_path / "processed"
    jobs_dir.mkdir()
    processed_dir.mkdir()

    job_file = jobs_dir / "inline_amp.txt"
    job_file.write_text("echo hello &\npython train.py --x 1\n")

    io = m.TaskIO(jobs_dir=str(jobs_dir), processed_dir=str(processed_dir))
    asyncio.run(io.sync_file())

    popped = {io.task_queue.get_nowait(), io.task_queue.get_nowait()}
    assert popped == {"echo hello &", "python train.py --x 1"}


async def _run_worker_once(worker):
    loop = asyncio.create_task(worker.schedule())
    await asyncio.sleep(0.25)
    loop.cancel()
    await asyncio.gather(loop, return_exceptions=True)


def test_worker_retries_nonzero_exit_and_requeues_tail(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    # Keep scheduler out of idle-path dispatch so queue order assertions stay stable.
    m.get_system_memory_usage = lambda: 85.0
    m.available_gpus = [0]

    worker = m.ProcessorWorker(0, max_retries=3)
    worker.task_queue = m.HybridTaskQueue()
    worker.message_queue = asyncio.Queue()
    worker.task_pool = defaultdict(set)
    worker.task_pool_lock = asyncio.Lock()

    worker.task_queue.put_nowait("tail_task")
    worker.task_pool["running"].add("task_fail")
    worker.running_proc["task_fail"] = _FakeProc(returncode=2)
    worker.task_start_times["task_fail"] = time.time()

    asyncio.run(_run_worker_once(worker))

    assert worker.retry_counts["task_fail"] == 1
    assert "task_fail" in worker.task_pool["waiting"]
    assert worker.task_queue.get_nowait() == "tail_task"
    assert worker.task_queue.get_nowait() == "task_fail"
    assert worker.message_queue.get_nowait() == ("task_fail", "waiting")


def test_worker_stops_retry_after_limit(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    m.get_system_memory_usage = lambda: 85.0
    m.available_gpus = [0]

    worker = m.ProcessorWorker(0, max_retries=1)
    worker.task_queue = m.HybridTaskQueue()
    worker.message_queue = asyncio.Queue()
    worker.task_pool = defaultdict(set)
    worker.task_pool_lock = asyncio.Lock()

    worker.task_pool["running"].add("task_fail")
    worker.retry_counts["task_fail"] = 1
    worker.running_proc["task_fail"] = _FakeProc(returncode=9)
    worker.task_start_times["task_fail"] = time.time()

    asyncio.run(_run_worker_once(worker))

    assert "task_fail" not in worker.retry_counts
    assert worker.message_queue.get_nowait() == ("task_fail", "terminated")
    assert worker.task_queue.qsize() == 0


def test_worker_clears_retry_count_on_success(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    m.get_system_memory_usage = lambda: 85.0
    m.available_gpus = [0]

    worker = m.ProcessorWorker(0, max_retries=3)
    worker.task_queue = m.HybridTaskQueue()
    worker.message_queue = asyncio.Queue()
    worker.task_pool = defaultdict(set)
    worker.task_pool_lock = asyncio.Lock()

    worker.task_pool["running"].add("task_ok")
    worker.retry_counts["task_ok"] = 2
    worker.running_proc["task_ok"] = _FakeProc(returncode=0)
    worker.task_start_times["task_ok"] = time.time()

    asyncio.run(_run_worker_once(worker))

    assert "task_ok" not in worker.retry_counts
    assert worker.message_queue.get_nowait() == ("task_ok", "finished")


def test_worker_gpu_removed_interrupts_and_requeues_front(monkeypatch, tmp_path):
    m = _load_async_train_module(monkeypatch, tmp_path)
    m.get_system_memory_usage = lambda: 10.0
    m.available_gpus = []

    worker = m.ProcessorWorker(0, max_retries=3)
    worker.task_queue = m.HybridTaskQueue()
    worker.message_queue = asyncio.Queue()
    worker.task_pool = defaultdict(set)
    worker.task_pool_lock = asyncio.Lock()

    worker.task_queue.put_nowait("normal_task")
    worker.task_pool["running"].add("task_on_removed_gpu")
    proc = _FakeProc(returncode=None)
    worker.running_proc["task_on_removed_gpu"] = proc
    worker.task_start_times["task_on_removed_gpu"] = time.time()

    asyncio.run(_run_worker_once(worker))

    assert proc.terminated
    assert "task_on_removed_gpu" in worker.task_pool["waiting"]
    assert "task_on_removed_gpu" not in worker.task_pool["running"]
    assert worker.task_queue.get_nowait() == "task_on_removed_gpu"
    assert worker.task_queue.get_nowait() == "normal_task"
    assert worker.message_queue.get_nowait() == ("task_on_removed_gpu", "waiting")
