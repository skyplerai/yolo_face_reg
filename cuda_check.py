import dlib
import torch
import cv2

def check_dlib_device():
    try:
        use_cuda = dlib.DLIB_USE_CUDA
        print(f"dlib version: {dlib.__version__}")
        print(f"dlib CUDA support: {use_cuda}")
    except AttributeError:
        print("dlib CUDA support: False")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    check_dlib_device()
import torch
print(torch.cuda.is_available())
