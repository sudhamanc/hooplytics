"""
Training Script for NBA Player Tier Classifier
This script trains the neural network model to classify players into performance tiers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from player_classifier_model import create_model, TierLabels


class PlayerDataset(Dataset):
    """PyTorch Dataset for NBA player statistics."""
    
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X: Feature array (numpy array)
            y: Label array (numpy array)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PlayerClassifierTrainer:
    """Trainer class for the player classification model."""
    
    def __init__(self, data_dir=None, model_dir=None, device=None):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing preprocessed data
            model_dir: Directory to save trained models
            device: Device to use (cuda/cpu), auto-detected if None
        """
        # Use paths relative to this script's location
        script_dir = Path(__file__).resolve().parent
        if data_dir is None:
            self.data_dir = script_dir.parent / 'data'
        else:
            self.data_dir = Path(data_dir).resolve()
        
        if model_dir is None:
            self.model_dir = script_dir.parent / 'data' / 'models'
        else:
            self.model_dir = Path(model_dir).resolve()
        
        self.model_dir.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def load_data(self):
        """
        Load preprocessed data from disk.
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        print("Loading preprocessed data...")
        
        X = np.load(self.data_dir / 'X_train.npy')
        y = np.load(self.data_dir / 'y_train.npy')
        
        with open(self.data_dir / 'scaler_params.json', 'r') as f:
            scaler_params = json.load(f)
            feature_names = scaler_params['feature_names']
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y, feature_names
    
    def prepare_dataloaders(self, X, y, train_split=0.8, batch_size=32):
        """
        Prepare training and validation dataloaders.
        
        Args:
            X: Feature array
            y: Label array
            train_split: Proportion of data for training
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print(f"Preparing dataloaders (train_split={train_split}, batch_size={batch_size})...")
        
        # Create dataset
        dataset = PlayerDataset(X, y)
        
        # Split into train and validation
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """
        Train for one epoch.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion):
        """
        Validate the model.
        
        Args:
            model: Neural network model
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs=100, learning_rate=0.001, batch_size=32, patience=15):
        """
        Train the model with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Trained model
        """
        print("="*60)
        print("Starting Training")
        print("="*60)
        
        # Load data
        X, y, feature_names = self.load_data()
        
        # Prepare dataloaders
        train_loader, val_loader = self.prepare_dataloaders(X, y, batch_size=batch_size)
        
        # Create model
        model = create_model(input_size=X.shape[1])
        model = model.to(self.device)
        
        print("\nModel Architecture:")
        info = model.get_model_info()
        for layer in info['layers']:
            print(f"  {layer}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Early Stopping Patience: {patience}")
        print(f"  Device: {self.device}")
        print("\n" + "="*60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print("="*60)
        
        return model
    
    def evaluate(self, model, X, y):
        """
        Evaluate the model and generate metrics.
        
        Args:
            model: Trained model
            X: Features
            y: Labels
        """
        print("\nEvaluating model...")
        
        model.eval()
        dataset = PlayerDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Classification report
        print("\nClassification Report:")
        # Get unique labels present in the data
        unique_labels = sorted(set(all_labels))
        present_tier_names = [TierLabels.TIER_NAMES[i] for i in unique_labels]
        
        print(classification_report(
            all_labels, 
            all_preds, 
            labels=unique_labels,
            target_names=present_tier_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=present_tier_names,
            yticklabels=present_tier_names
        )
        plt.title('Confusion Matrix - Player Tier Classification')
        plt.ylabel('True Tier')
        plt.xlabel('Predicted Tier')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'confusion_matrix.png', dpi=150)
        print(f"\nConfusion matrix saved to {self.model_dir / 'confusion_matrix.png'}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=150)
        print(f"Training history saved to {self.model_dir / 'training_history.png'}")
    
    def save_model(self, model, feature_names):
        """
        Save the trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        model_path = self.model_dir / 'player_classifier.pth'
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_params': {
                'input_size': model.input_size,
                'hidden_sizes': model.hidden_sizes,
                'num_classes': model.num_classes,
                'dropout_rate': model.dropout_rate
            },
            'feature_names': feature_names,
            'tier_names': TierLabels.TIER_NAMES,
            'training_history': self.history
        }
        
        torch.save(checkpoint, model_path)
        print(f"\nModel saved to {model_path}")


def main():
    """Main function to run the training pipeline."""
    # Initialize trainer
    trainer = PlayerClassifierTrainer()
    
    # Train model
    model = trainer.train(
        num_epochs=100,
        learning_rate=0.001,
        batch_size=32,
        patience=15
    )
    
    # Load data for evaluation
    X, y, feature_names = trainer.load_data()
    
    # Evaluate model
    trainer.evaluate(model, X, y)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model(model, feature_names)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
