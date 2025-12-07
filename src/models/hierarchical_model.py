"""
Hierarchical Model for Group Activity Recognition

Combines Person LSTM (Stage 1) and Group LSTM (Stage 2)
into a complete two-stage hierarchical model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class HierarchicalModel(nn.Module):
    """
    Complete two-stage hierarchical model.
    
    Stage 1: Person LSTM processes each person's temporal features
    Stage 2: Group LSTM processes pooled person features to predict activity
    """
    
    def __init__(
        self,
        feature_dim: int = 9216,  # AlexNet produces 9216, not 4096!
        person_hidden_dim: int = 3000,
        group_hidden_dim: int = 500,
        num_action_classes: int = 9,
        num_activity_classes: int = 8,
        pooling: str = "max",
        person_num_layers: int = 1,
        group_num_layers: int = 1,
        dropout: float = 0.3
    ):
        """
        Initialize hierarchical model.
        
        Args:
            feature_dim: Dimension of input CNN features (9216 for AlexNet)
            person_hidden_dim: Hidden dimension for person LSTM (3000)
            group_hidden_dim: Hidden dimension for group LSTM (500)
            num_action_classes: Number of individual action classes (9)
            num_activity_classes: Number of group activity classes (8)
            pooling: Pooling strategy ('max', 'avg', or 'attention')
            person_num_layers: Number of layers in person LSTM
            group_num_layers: Number of layers in group LSTM
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.person_hidden_dim = person_hidden_dim
        self.group_hidden_dim = group_hidden_dim
        self.pooling = pooling
        
        # Stage 1: Person LSTM
        self.person_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=person_hidden_dim,
            num_layers=person_num_layers,
            batch_first=True,
            dropout=dropout if person_num_layers > 1 else 0
        )
        
        # Optional: Action classifier for multi-task learning
        self.action_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(person_hidden_dim, num_action_classes)
        )
        
        # Concatenate CNN features + Person LSTM hidden states
        concat_dim = feature_dim + person_hidden_dim
        
        # Stage 2: Group LSTM
        # Input layer
        self.group_fc_input = nn.Sequential(
            nn.Linear(concat_dim, group_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM
        self.group_lstm = nn.LSTM(
            input_size=group_hidden_dim,
            hidden_size=group_hidden_dim,
            num_layers=group_num_layers,
            batch_first=True,
            dropout=dropout if group_num_layers > 1 else 0
        )
        
        # Activity classifier
        self.activity_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(group_hidden_dim, num_activity_classes)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_person_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model.
        
        Args:
            features: Input features (batch, seq_len, num_players, feature_dim)
            return_person_features: Whether to return person-level features
            
        Returns:
            Dictionary containing:
                - activity_logits: (batch, num_activity_classes)
                - action_logits: (batch, seq_len, num_players, num_action_classes)
                - person_features: (optional) person-level LSTM outputs
        """
        batch_size, seq_len, num_players, feat_dim = features.shape
        
        # Stage 1: Process each person independently
        # Reshape to (batch * num_players, seq_len, feature_dim)
        features_flat = features.view(batch_size * num_players, seq_len, feat_dim)
        
        # Apply person LSTM
        person_lstm_out, _ = self.person_lstm(features_flat)
        # person_lstm_out: (batch * num_players, seq_len, person_hidden_dim)
        
        # Reshape back to (batch, num_players, seq_len, person_hidden_dim)
        person_hidden = person_lstm_out.view(
            batch_size, num_players, seq_len, self.person_hidden_dim
        )
        # Transpose to (batch, seq_len, num_players, person_hidden_dim)
        person_hidden = person_hidden.transpose(1, 2)
        
        # Concatenate original features with person hidden states
        # features: (batch, seq_len, num_players, feature_dim)
        # person_hidden: (batch, seq_len, num_players, person_hidden_dim)
        person_features = torch.cat([features, person_hidden], dim=-1)
        # person_features: (batch, seq_len, num_players, feature_dim + person_hidden_dim)
        
        # Pool across players at each timestep
        if self.pooling == "max":
            pooled_features, _ = torch.max(person_features, dim=2)
        elif self.pooling == "avg":
            pooled_features = torch.mean(person_features, dim=2)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # pooled_features: (batch, seq_len, feature_dim + person_hidden_dim)
        
        # Stage 2: Process pooled features with group LSTM
        # Project input
        x = self.group_fc_input(pooled_features)
        
        # LSTM
        group_lstm_out, (h_n, c_n) = self.group_lstm(x)
        
        # Use last hidden state for activity classification
        last_hidden = h_n[-1]  # (batch, group_hidden_dim)
        activity_logits = self.activity_classifier(last_hidden)
        
        # Optional: Compute action logits for multi-task learning
        # Use person_hidden for action classification
        action_logits = self.action_classifier(person_hidden)
        # action_logits: (batch, seq_len, num_players, num_action_classes)
        
        outputs = {
            'activity_logits': activity_logits,
            'action_logits': action_logits
        }
        
        if return_person_features:
            outputs['person_features'] = person_features
        
        return outputs
    
    def freeze_person_lstm(self):
        """Freeze person LSTM parameters (for stage 2 training)"""
        for param in self.person_lstm.parameters():
            param.requires_grad = False
        for param in self.action_classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_person_lstm(self):
        """Unfreeze person LSTM parameters"""
        for param in self.person_lstm.parameters():
            param.requires_grad = True
        for param in self.action_classifier.parameters():
            param.requires_grad = True


# Test the model
if __name__ == "__main__":
    print("Testing HierarchicalModel...")
    
    # Create model
    model = HierarchicalModel(
        feature_dim=9216,  # AlexNet features
        person_hidden_dim=3000,
        group_hidden_dim=500,
        num_action_classes=9,
        num_activity_classes=8,
        pooling='max'
    )
    
    # Test input
    batch_size = 4
    seq_len = 10
    num_players = 12
    feature_dim = 9216
    
    x = torch.randn(batch_size, seq_len, num_players, feature_dim)
    
    # Forward pass
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Activity logits shape: {outputs['activity_logits'].shape}")
    print(f"Action logits shape: {outputs['action_logits'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test freezing
    print("\nTesting parameter freezing...")
    model.freeze_person_lstm()
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after freeze: {trainable_after_freeze:,}")
    
    model.unfreeze_person_lstm()
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable_after_unfreeze:,}")
    
    print("\n✓ HierarchicalModel test passed!")