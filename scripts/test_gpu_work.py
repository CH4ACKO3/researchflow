#!/usr/bin/env python3
"""
Test GPU workload script for async_train.py testing.
This script occupies GPU memory and compute for a random duration.
"""
import torch
import time
import random
import sys
import argparse
import logging

def setup_logger(task_id):
    """Setup logger for this task"""
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - TASK_%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def main():
    parser = argparse.ArgumentParser(description='Test GPU workload')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (if not provided, random between 2-10)')
    parser.add_argument('--task-id', type=str, default=None,
                        help='Task ID for logging')
    parser.add_argument('--memory-percent', type=float, default=None,
                        help='GPU memory usage percentage (e.g., 30 for 30%%)')
    args = parser.parse_args()
    
    # Get duration
    if args.duration is not None:
        duration = args.duration
    else:
        duration = random.uniform(2.0, 10.0)
    
    # Setup logger
    task_id = args.task_id if args.task_id else f"unknown_{int(time.time())}"
    logger = setup_logger(task_id)
    
    logger.info(f"Starting task {task_id}, duration: {duration:.2f}s")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    logger.info(f"Using device: {device}")
    
    # Calculate memory to allocate
    if args.memory_percent is not None:
        # Get total GPU memory
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        target_memory_gb = total_memory_gb * (args.memory_percent / 100.0)
        memory_mb = int(target_memory_gb * 1024)
        logger.info(f"Target memory: {args.memory_percent}%% of {total_memory_gb:.2f}GB = {target_memory_gb:.2f}GB ({memory_mb}MB)")
    else:
        # Default: random between 500MB - 2GB
        memory_mb = random.randint(500, 2000)
        logger.info(f"Allocating {memory_mb}MB GPU memory (random)")
    
    # Create tensors to occupy memory
    tensors = []
    try:
        # Allocate memory in chunks
        chunk_size = 100 * 1024 * 1024  # 100MB chunks
        num_chunks = memory_mb // 100
        
        for i in range(num_chunks):
            tensor = torch.randn(chunk_size // 4, device=device)  # 4 bytes per float32
            tensors.append(tensor)
            if i % 5 == 0:
                logger.info(f"Allocated chunk {i+1}/{num_chunks}")
        
        logger.info(f"Memory allocation complete, starting computation")
        
        # Keep GPU busy with computations
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            # Perform matrix operations to keep GPU busy
            if tensors:
                # Matrix multiplication to utilize GPU compute
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.matmul(a, b)
                # Use the result to prevent optimization
                _ = c.sum()
            
            iteration += 1
            # Small sleep to prevent overheating
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        logger.info(f"Task {task_id} completed successfully after {elapsed:.2f}s, iterations: {iteration}")
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

