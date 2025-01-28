import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# check and set device to run on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_data():
    """Load and normalize the MNIST data"""
    print("Loading MNIST data")
    # define how data should be transformed
    transform = transforms.Compose([
        transforms.ToTensor(), 
        ])
    # load data
    train_dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Print the sizes of the datasets
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    return train_loader, test_loader

def check_transformation():
    """Check that the pixel values are actually between 0 and 1"""
    print("Verifying normalization")
    try:
        # Load data and get a batch
        train_loader, test_loader = load_data()
        data_iter = iter(train_loader)
        images, labels = next(data_iter)

        # Check and print the min and max pixel values for the first batch
        print(f"Min pixel value: {images.min().item()}")
        print(f"Max pixel value: {images.max().item()}")
    except Exception as e:
        print(f"Pixels not between 0 and 1: {e}")


if __name__ == "__main__":
    load_data()
    check_transformation()