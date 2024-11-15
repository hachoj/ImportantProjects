import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.transforms = transforms
        self.num_global_crops = 2
        self.num_local_crops = 6
        self.global_transform1 = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), # Ensure RGB format
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.global_transform2 = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), # Ensure RGB format
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.local_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), # Ensure RGB format
            transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),  # Smaller, more concentrated crops
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Same level of jittering
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        labels = item["label"]

        global_view1 = self.global_transform1(image)
        global_view2 = self.global_transform2(image)
        global_views = [global_view1, global_view2]

        local_views = [self.local_transform(image) for _ in range(self.num_local_crops)]

        views = global_views + local_views

        return views, labels

def custom_collate(batch):
    """
    Custom collate function to handle multiple views efficiently
    Args:
        batch: List of tuples (views, label) from __getitem__
    Returns:
        Tuple of (stacked_views, stacked_labels)
    """
    # Separate views and labels
    views, labels = zip(*batch)
    
    # Stack all global views together and all local views together
    global_views = torch.stack([torch.stack([sample[0], sample[1]]) for sample in views])  # Shape: [batch_size, 2, C, H, W]
    local_views = torch.stack([torch.stack(sample[2:]) for sample in views])   # Shape: [batch_size, 6, C, H, W]
    
    # Stack labels
    labels = torch.tensor(labels)
    
    return (global_views, local_views), labels

class TinyImageNetDataLoader: 
    def __init__(self, batch_size=32, num_workers=4):
        # Load the Tiny-ImageNet dataset using Hugging Face datasets
        dataset = load_dataset("zh-plus/tiny-imagenet")
        # Create custom Dataset instances for train and validation
        self.train_data = TinyImageNetDataset(hf_dataset=dataset["train"])
        self.valid_data = TinyImageNetDataset(hf_dataset=dataset["valid"])

        # Initialize DataLoaders for train and validation datasets
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,  # Shuffling is generally enabled for training
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate
        )
        self.valid_loader = DataLoader(
            self.valid_data,
            batch_size=batch_size,
            shuffle=False,  # Shuffling usually disabled for validation
            num_workers=num_workers,
            pin_memory=True,
        )
