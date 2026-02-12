import pytest
import torch
import gc

@pytest.fixture(scope='session', autouse=True)
def check_cuda_memory():
    """Check for memory leaks across the entire test session"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated()
        yield
        
        # Cleanup before final check
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated()
        
        # Allow for 30MB overhead across all tests (CUDA runtime keeps some memory)
        memory_diff = final_memory - initial_memory
        assert memory_diff < 30e6, f"Memory leak detected: {memory_diff/1e6:.1f}MB leaked across session"
    else:
        yield

@pytest.fixture(autouse=True)
def cleanup_between_tests():
    """Clean up GPU memory between each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()