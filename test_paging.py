import os
import mmap
import time
import psutil
import torch
from llama_cpp import Llama
import argparse

def measure_rss():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def test_paged_loading(model_path):
    print(f"[N] testing paged loading for: {model_path}")
    
    start_rss = measure_rss()
    print(f"[N] start RSS: {start_rss:.2f} MB")
    
    # Load with mmap=True (default) but try to avoid prefaulting
    # llama-cpp-python usually uses mmap if possible
    print("[N] loading model with mmap...")
    start_time = time.time()
    model = Llama(
        model_path=model_path,
        n_ctx=512,
        use_mmap=True,
        use_mlock=False, # Crucial: don't lock in RAM
        verbose=False
    )
    load_time = time.time() - start_time
    
    after_load_rss = measure_rss()
    print(f"[N] RSS after load: {after_load_rss:.2f} MB (Delta: {after_load_rss - start_rss:.2f} MB)")
    print(f"[N] load time: {load_time:.2f}s")
    
    # Run a simple query to trigger page faults for necessary tensors
    print("[N] running inference to trigger paging...")
    start_time = time.time()
    output = model("What is 2+2?", max_tokens=10)
    inference_time = time.time() - start_time
    
    after_inference_rss = measure_rss()
    print(f"[N] RSS after inference: {after_inference_rss:.2f} MB (Delta: {after_inference_rss - after_load_rss:.2f} MB)")
    print(f"[N] total RSS increase: {after_inference_rss - start_rss:.2f} MB")
    print(f"[N] inference time: {inference_time:.2f}s")
    
    # Check if RSS is significantly smaller than model file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[N] model file size: {file_size:.2f} MB")
    
    reduction = 1 - (after_inference_rss - start_rss) / file_size
    print(f"[N] effective RAM reduction: {reduction*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    test_paged_loading(args.model)
