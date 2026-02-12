import pytest
import torch
import gc

def test_vram_monitor():
    """Test VRAM monitoring and cleanup"""
    torch.cuda.empty_cache()
    gc.collect()
    
    initial_memory = torch.cuda.memory_allocated()
    
    # Simulate memory usage with a smaller tensor to avoid excessive allocation
    tensor = torch.randn(5000, 5000, device='cuda')
    current_memory = torch.cuda.memory_allocated()
    
    assert current_memory > initial_memory, "Tensor allocation should increase memory"
    
    # Release memory
    del tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    final_memory = torch.cuda.memory_allocated()
    
    # Allow for CUDA memory allocator overhead (allocates in larger chunks)
    memory_diff = final_memory - initial_memory
    assert memory_diff < 2e6, f"Memory not properly released: {memory_diff/1e6:.2f}MB still allocated"