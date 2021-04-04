import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Cuda is enabled")
        print("Using " + torch.cuda.get_device_name())
    else:
        print("Not using coda")
