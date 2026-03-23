"""Data pipeline and training utilities for low-power computer vision."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EdgeVisionDataset(Dataset):
    """Dataset for edge computer vision tasks."""
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
        target_classes: Optional[List[int]] = None,
    ):
        """Initialize dataset.
        
        Args:
            images: Image data array
            labels: Label array
            transform: Optional image transforms
            target_classes: Specific classes to keep (None for all)
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_classes = target_classes
        
        if target_classes is not None:
            self._filter_classes()
    
    def _filter_classes(self) -> None:
        """Filter dataset to only include target classes."""
        mask = np.isin(self.labels, self.target_classes)
        self.images = self.images[mask]
        self.labels = self.labels[mask]
        
        # Remap labels to 0, 1, 2, ...
        label_map = {cls: idx for idx, cls in enumerate(self.target_classes)}
        self.labels = np.array([label_map[label] for label in self.labels])
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image, label)
        """
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # Convert to PIL Image if needed
        if not isinstance(image, torch.Tensor):
            from PIL import Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EdgeTrainer:
    """Trainer for edge computer vision models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Union[str, float, int]],
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device for computation
            config: Training configuration
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get("optimizer", "adam")
        learning_rate = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name.lower() == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", None)
        
        if scheduler_name is None:
            return None
        
        if scheduler_name.lower() == "step":
            step_size = self.config.get("step_size", 30)
            gamma = self.config.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        elif scheduler_name.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 100),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.get("epochs", 100)
        
        logging.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                
                logging.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )
            else:
                logging.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%"
                )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        return self.history
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            class_names: Class names for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        results = {
            "accuracy": accuracy,
            "predictions": np.array(all_predictions),
            "targets": np.array(all_targets),
        }
        
        if class_names is not None:
            results["classification_report"] = classification_report(
                all_targets, all_predictions, target_names=class_names
            )
            results["confusion_matrix"] = confusion_matrix(all_targets, all_predictions)
        
        return results


def create_data_loaders(
    dataset_name: str = "cifar10",
    target_classes: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of dataset to use
        target_classes: Specific classes to keep
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        data_dir: Directory to store data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if dataset_name.lower() == "cifar10":
        return _create_cifar10_loaders(
            target_classes, batch_size, num_workers, data_dir
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _create_cifar10_loaders(
    target_classes: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-10 data loaders."""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Filter classes if specified
    if target_classes is not None:
        train_dataset = EdgeVisionDataset(
            train_dataset.data,
            np.array(train_dataset.targets),
            transform=train_transform,
            target_classes=target_classes,
        )
        test_dataset = EdgeVisionDataset(
            test_dataset.data,
            np.array(test_dataset.targets),
            transform=test_transform,
            target_classes=target_classes,
        )
    
    # Split training data into train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history["train_acc"], label="Train Accuracy")
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
