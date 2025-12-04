"""
Neural Network Model for NBA Player Tier Classification
This module defines the architecture for classifying players into performance tiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayerClassifierNN(nn.Module):
    """
    Neural Network for classifying NBA players into performance tiers.
    
    Architecture:
    - Input Layer: 13 features (player statistics)
    - Hidden Layer 1: 64 neurons with ReLU and Dropout
    - Hidden Layer 2: 32 neurons with ReLU and Dropout
    - Hidden Layer 3: 16 neurons with ReLU and Dropout
    - Output Layer: 5 neurons (5 tier classes) with Softmax
    
    Tiers:
    - 0: Bench
    - 1: Rotation
    - 2: Starter
    - 3: All-Star
    - 4: Elite
    """
    
    def __init__(self, input_size=13, hidden_sizes=[64, 32, 16], num_classes=5, dropout_rate=0.3):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes (tiers)
            dropout_rate: Dropout rate for regularization
        """
        super(PlayerClassifierNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Input layer -> Hidden layer 1
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layer 1 -> Hidden layer 2
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Hidden layer 2 -> Hidden layer 3
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Hidden layer 3 -> Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc4(x)
        
        return x
    
    def predict(self, x):
        """
        Make predictions with probability scores.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities
    
    def get_model_info(self):
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'PlayerClassifierNN',
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': [
                f'Input: {self.input_size}',
                f'Hidden1: {self.hidden_sizes[0]} (ReLU + BatchNorm + Dropout)',
                f'Hidden2: {self.hidden_sizes[1]} (ReLU + BatchNorm + Dropout)',
                f'Hidden3: {self.hidden_sizes[2]} (ReLU + BatchNorm + Dropout)',
                f'Output: {self.num_classes} (Linear)'
            ]
        }


class TierLabels:
    """Helper class for tier label names and descriptions."""
    
    TIER_NAMES = ['Bench', 'Rotation', 'Starter', 'All-Star', 'Elite']
    
    TIER_DESCRIPTIONS = {
        0: 'Bench Player - Limited minutes, developing skills',
        1: 'Rotation Player - Regular contributor off the bench',
        2: 'Starter - Consistent starter with solid production',
        3: 'All-Star - Top-tier player, all-star caliber',
        4: 'Elite - Superstar, MVP candidate'
    }
    
    TIER_COLORS = {
        0: '#9CA3AF',  # Gray
        1: '#3B82F6',  # Blue
        2: '#10B981',  # Green
        3: '#F59E0B',  # Amber
        4: '#EF4444'   # Red
    }
    
    @classmethod
    def get_tier_name(cls, tier_id):
        """Get the name of a tier."""
        return cls.TIER_NAMES[tier_id] if 0 <= tier_id < len(cls.TIER_NAMES) else 'Unknown'
    
    @classmethod
    def get_tier_description(cls, tier_id):
        """Get the description of a tier."""
        return cls.TIER_DESCRIPTIONS.get(tier_id, 'Unknown tier')
    
    @classmethod
    def get_tier_color(cls, tier_id):
        """Get the color code for a tier."""
        return cls.TIER_COLORS.get(tier_id, '#000000')


def create_model(input_size=13, hidden_sizes=[64, 32, 16], num_classes=5, dropout_rate=0.3):
    """
    Factory function to create a new model instance.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        PlayerClassifierNN instance
    """
    model = PlayerClassifierNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model


def load_model(model_path, device='cpu'):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model instance
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model hyperparameters
    model_params = checkpoint.get('model_params', {})
    input_size = model_params.get('input_size', 13)
    hidden_sizes = model_params.get('hidden_sizes', [64, 32, 16])
    num_classes = model_params.get('num_classes', 5)
    dropout_rate = model_params.get('dropout_rate', 0.3)
    
    # Create model instance
    model = create_model(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing PlayerClassifierNN model...")
    print("="*60)
    
    # Create model
    model = create_model()
    print("\nModel created successfully!")
    
    # Display model info
    info = model.get_model_info()
    print("\nModel Information:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print("\nLayer Architecture:")
    for layer in info['layers']:
        print(f"  {layer}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 10
    input_size = 13
    test_input = torch.randn(batch_size, input_size)
    
    output = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test prediction
    predicted_class, probabilities = model.predict(test_input)
    print(f"\nPrediction test:")
    print(f"  Predicted classes: {predicted_class}")
    print(f"  Probabilities shape: {probabilities.shape}")
    
    # Display tier information
    print("\nTier Information:")
    for i, name in enumerate(TierLabels.TIER_NAMES):
        print(f"  Tier {i}: {name} - {TierLabels.get_tier_description(i)}")
    
    print("\n" + "="*60)
    print("Model test completed successfully!")
