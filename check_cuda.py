"""Check CUDA availability for PyTorch."""

import torch


def main() -> None:
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        return

    print(f"CUDA device count: {torch.cuda.device_count()}")
    current = torch.cuda.current_device()
    name = torch.cuda.get_device_name(current)
    print(f"Current CUDA device: {current}")
    print(f"Device name: {name}")


if __name__ == "__main__":
    main()
