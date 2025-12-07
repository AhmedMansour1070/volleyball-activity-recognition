"""
Group LSTM - Stage 2 of Hierarchical Model

Models group activity dynamics using aggregated person features.
"""

import torch
import torch.nn as nn
from typing import Optional


class GroupLSTM(nn.Module):
    """
    Stage 2: LSTM model for group activity recognition.
    
    Takes pooled person-level features and models the temporal
    dynamics of the group activity.
    """
    
    def __init__(
        self,
        input_dim: int = 7096,  # 4096 (CNN) + 3000 (Person LSTM hidden)
        hidden_dim: int = 500,
        num_activity_classes: int = 8,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        """
        Initialize Group LSTM.
        
        Args:
            input_dim: Input dimension (pooled features)
            hidden_dim: LSTM hidden dimension (paper uses 500)
            num_activity_classes: Number of group activity classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_activity_classes = num_activity_classes
        
        # Fully connected layer before LSTM
        self.fc_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Activity classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_activity_classes)
        )
    
    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pooled_features: (batch, seq_len, input_dim)
            
        Returns:
            logits: (batch, num_activity_classes)
        """
        # Project input
        x = self.fc_input(pooled_features)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for classification
        # h_n shape: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Classify
        logits = self.classifier(last_hidden)
        
        return logits


# Test the model
if __name__ == "__main__":
    print("Testing GroupLSTM...")
    
    # Create model
    model = GroupLSTM(
        input_dim=7096,  # 4096 + 3000
        hidden_dim=500,
        num_activity_classes=8,
        num_layers=1,
        dropout=0.3
    )
    
    # Test input (pooled features)
    batch_size = 4
    seq_len = 10
    input_dim = 7096
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ GroupLSTM test passed!")