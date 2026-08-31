import sys
import torch
import whisper
import openpyxl
import shutil


def check():
    print("CHECKING INSTALLS")
    print("Python version:", sys.version)
    print("Whisper version: ", whisper.__version__)
    print("openpyxl version: ", openpyxl.__version__)
    print("PyTorch version: ", torch.__version__)

    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print("FFmpeg: OK")
        print("FFmpeg location:", ffmpeg)
    else:
        print("FFmpeg: NOT FOUND")
    print("DONE")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    check()

