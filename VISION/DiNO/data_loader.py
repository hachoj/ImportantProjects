import torchvision
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None):
        self.dataset = hf_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if self.transforms:
            image = self.transforms(image)

        return image, label

class TinyImageNetDataLoader:
    def __init__(self, batch_size=64, num_workers=4):
        # Load the Tiny-ImageNet dataset using Hugging Face datasets
        dataset = load_dataset("zh-plus/tiny-imagenet")

        # Define the transformations
        self.train_transforms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert grayscale to RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self.valid_transforms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert grayscale to RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Create custom Dataset instances for train and validation
        self.train_data = TinyImageNetDataset(
            hf_dataset=dataset['train'],
            transforms=self.train_transforms
        )
        self.valid_data = TinyImageNetDataset(
            hf_dataset=dataset['valid'],
            transforms=self.valid_transforms
        )

        # Initialize DataLoaders for train and validation datasets
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,  # Shuffling is generally enabled for training
            num_workers=num_workers,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            self.valid_data,
            batch_size=batch_size,
            shuffle=False,  # Shuffling usually disabled for validation
            num_workers=num_workers,
            pin_memory=True
        )