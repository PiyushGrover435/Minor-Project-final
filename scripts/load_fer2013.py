import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Expected classes by folder name (matching our map):
# angry, disgust, fear, happy, neutral, sad, surprise

def get_fer2013_dataloaders(base_dir, batch_size=64, num_workers=2):
    """
    Creates train and test DataLoaders for FER-2013 using ImageFolder.
    Assumes standard train/test split folders.
    """
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    # FER-2013 is 48x48 grayscale. 
    # Usually saved as RGB locally, we'll ensure Grayscale.
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Store class_to_idx to be able to map back
    class_to_idx = train_dataset.class_to_idx
    
    return train_loader, test_loader, class_to_idx

if __name__ == "__main__":
    base = os.path.join("Dataset", "FER-2013")
    train_loader, test_loader, class_idx = get_fer2013_dataloaders(base, batch_size=32, num_workers=0)
    
    print("Class mapping:", class_idx)
    for images, labels in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch labels shape:", labels.shape)
        break
    print(f"Num train batches: {len(train_loader)}")
    print(f"Num test batches: {len(test_loader)}")
