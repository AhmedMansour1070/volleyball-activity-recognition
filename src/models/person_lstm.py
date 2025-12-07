"""
Person LSTM - Stage 1 of Hierarchical Model

Models individual person action dynamics using LSTM.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PersonLSTM(nn.Module):
    """
    Stage 1: LSTM model for individual person actions.
    
    Processes temporal features for each person independently
    to capture their action dynamics over time.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 3000,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Initialize Person LSTM.
        
        Args:
            input_dim: Dimension of input features (e.g., 4096 for AlexNet fc7)
            hidden_dim: Hidden dimension of LSTM (paper uses 3000)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Input features (batch, seq_len, input_dim)
            hidden: Initial hidden state (optional)
            
        Returns:
            lstm_out: LSTM outputs (batch, seq_len, hidden_dim)
            hidden: Final hidden state tuple (h_n, c_n)
        """
        lstm_out, hidden = self.lstm(features, hidden)
        return lstm_out, hidden


class PersonLSTMWithClassifier(nn.Module):
    """
    Person LSTM with action classification head.
    
    Used for Stage 1 pre-training on individual action labels.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 3000,
        num_action_classes: int = 9,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        """
        Initialize Person LSTM with classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: LSTM hidden dimension
            num_action_classes: Number of action classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.lstm = PersonLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # Action classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_action_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with classification.
        
        Args:
            features: (batch, seq_len, input_dim)
            
        Returns:
            logits: (batch, seq_len, num_action_classes)
        """
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)
        return logits


# Test the model
if __name__ == "__main__":
    print("Testing PersonLSTM...")
    
    # Create model
    model = PersonLSTM(
        input_dim=4096,
        hidden_dim=3000,
        num_layers=1,
        dropout=0.3
    )
    
    # Test input
    batch_size = 4
    seq_len = 10
    input_dim = 4096
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output, (h_n, c_n) = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {h_n.shape}")
    print(f"Cell state shape: {c_n.shape}")
    
    # Test with classifier
    print("\nTesting PersonLSTMWithClassifier...")
    
    model_with_classifier = PersonLSTMWithClassifier(
        input_dim=4096,
        hidden_dim=3000,
        num_action_classes=9
    )
    
    logits = model_with_classifier(x)
    print(f"Logits shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_with_classifier.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ PersonLSTM test passed!")