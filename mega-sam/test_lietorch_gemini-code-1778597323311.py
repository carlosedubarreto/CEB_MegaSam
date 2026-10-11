import torch
from lietorch import SO3

def test_lietorch():
    print("--- Starting LieTorch Installation Test ---")
    
    # 1. Basic Import Check
    try:
        import lietorch
        print(f"[OK] LieTorch imported successfully. Version: {getattr(lietorch, '__version__', 'N/A')}")
    except ImportError:
        print("[ERROR] LieTorch is not installed in this environment.")
        return

    # 2. CUDA Check
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available. LieTorch is designed for GPU use.")
        return
    print(f"[OK] CUDA detected: {torch.cuda.get_device_name(0)}")

    try:
        # 3. Test SO3 Exponential Map (Checks CUDA Kernel Execution)
        # Create random tangent space vectors
        phi = torch.randn(10, 3, device='cuda', requires_grad=True)
        
        # This triggers the custom CUDA 'exp' kernel
        R = SO3.exp(phi)
        print("[OK] Forward Pass: SO3.exp() successful.")

        # 4. Test Log Map & Autograd (Checks Backend Gradients)
        # This triggers the 'log' kernel and the backward graph
        loss = R.log().norm()
        loss.backward()
        
        if phi.grad is not None:
            print("[OK] Backward Pass: Gradients computed successfully.")
            print("\n[SUCCESS] LieTorch is installed and CUDA kernels are functional!")
        else:
            print("[ERROR] Backward pass finished but no gradients were computed.")

    except Exception as e:
        print(f"\n[ERROR] Testing failed with error: {e}")
        print("\nPossible solutions:")
        print("1. Check if your CUDA version matches the one used to build LieTorch.")
        print("2. Try reinstalling: pip install git+https://github.com/princeton-vl/lietorch.git")

if __name__ == "__main__":
    test_lietorch()