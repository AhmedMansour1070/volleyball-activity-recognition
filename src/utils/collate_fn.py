"""
Custom collate function for handling variable number of players
"""

import torch
from typing import List, Dict


def volleyball_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function that handles variable number of players.
    
    Pads all samples to have the same number of players (max in batch).
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary with padded tensors
    """
    # Find max number of players in this batch
    max_players = max(sample['features'].shape[1] for sample in batch)
    
    # Get dimensions
    seq_len = batch[0]['features'].shape[0]
    feature_dim = batch[0]['features'].shape[2]
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_features = torch.zeros(batch_size, seq_len, max_players, feature_dim)
    padded_action_labels = torch.full((batch_size, seq_len, max_players), -1, dtype=torch.long)
    activity_labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Lists for metadata
    video_ids = []
    frame_ids = []
    
    # Pad each sample
    for i, sample in enumerate(batch):
        num_players = sample['features'].shape[1]
        
        # Copy features
        padded_features[i, :, :num_players, :] = sample['features']
        
        # Copy action labels
        padded_action_labels[i, :, :num_players] = sample['action_labels']
        
        # Copy activity label
        activity_labels[i] = sample['activity_label']
        
        # Store metadata
        video_ids.append(sample['video_id'])
        frame_ids.append(sample['frame_id'])
    
    return {
        'features': padded_features,
        'activity_label': activity_labels,
        'action_labels': padded_action_labels,
        'video_id': video_ids,
        'frame_id': frame_ids,
        'num_players': torch.tensor([sample['features'].shape[1] for sample in batch])
    }