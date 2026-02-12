import pytest
import torch
import gc

def test_checkpoint_manager_memory_handling():
    """Test that memory is properly managed during training"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # Simulate a model and input data
    model = torch.nn.Linear(1000, 1000).cuda()
    input_data = torch.randn(64, 1000).cuda()

    # Check memory before training
    initial_memory = torch.cuda.memory_allocated()

    # Simulate training step
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    
    # Memory after backward pass (will be higher)
    after_backward_memory = torch.cuda.memory_allocated()

    # Cleanup - DELETE MODEL AND DATA
    del output, loss, input_data, model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check memory after cleanup
    final_memory = torch.cuda.memory_allocated()

    # Allow for CUDA overhead and model parameters (GPU memory allocator reserves chunks)
    # Linear(1000,1000) ≈ 8MB + allocator overhead
    memory_diff = final_memory - initial_memory
    assert memory_diff < 20e6, f"Memory leak: {memory_diff/1e6:.1f}MB not cleaned up"

def test_checkpoint_manager_no_cuda_out_of_memory():
    """Test that normal operations don't cause OOM"""
    torch.cuda.empty_cache()
    
    # Simulate a model and input data
    model = torch.nn.Linear(1000, 1000).cuda()
    input_data = torch.randn(64, 1000).cuda()

    try:
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        
        # Cleanup
        del output, loss, input_data, model
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            pytest.fail("CUDA out of memory error occurred during training.")
    finally:
        torch.cuda.empty_cache()