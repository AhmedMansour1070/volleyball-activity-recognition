"""
Feature Extraction Script for Volleyball Dataset

Extracts CNN features from player bounding boxes using pretrained models.
Saves features for later use in training the hierarchical LSTM model.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import yaml
from src.data.volleyball_loader import VolleyballDatasetLoader, VolleyballAnnotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.volleyball_loader import VolleyballDatasetLoader, VolleyballAnnotation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract CNN features from images"""
    
    def __init__(
        self,
        model_name: str = "alexnet",
        device: str = "cuda",
        image_size: int = 224
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: CNN model to use ('alexnet', 'resnet50', 'vgg16')
            device: Device to run on ('cuda' or 'cpu')
            image_size: Input image size
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        logger.info(f"Initializing {model_name} on {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        logger.info(f"Feature dimension: {self.feature_dim}")
    
    def _load_model(self) -> nn.Module:
        """Load pretrained model and remove classification head"""
        
        if self.model_name == "alexnet":
            # AlexNet (used in paper)
            model = models.alexnet(pretrained=True)
            # Remove classifier, keep features + avgpool
            model = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten()
            )
            
        elif self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            # Remove final FC layer
            model = nn.Sequential(*list(model.children())[:-1])
            model = nn.Sequential(model, nn.Flatten())
            
        elif self.model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten()
            )
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_feature_dim(self) -> int:
        """Determine output feature dimension"""
        dummy = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.no_grad():
            output = self.model(dummy)
        return output.shape[1]
    
    @torch.no_grad()
    def extract_from_image(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Feature vector as numpy array
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.model(img_tensor)
        
        return features.cpu().numpy()[0]
    
    @torch.no_grad()
    def extract_from_boxes(
        self,
        image: Image.Image,
        bboxes: List[tuple],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from multiple bounding boxes in an image.
        
        Args:
            image: PIL Image
            bboxes: List of (x, y, w, h) tuples
            batch_size: Batch size for processing
            
        Returns:
            Feature matrix (num_boxes, feature_dim)
        """
        if not bboxes:
            return np.zeros((0, self.feature_dim))
        
        # Crop all boxes
        crops = []
        for x, y, w, h in bboxes:
            # Convert to x1, y1, x2, y2
            x2 = x + w
            y2 = y + h
            
            # Crop and preprocess
            crop = image.crop((x, y, x2, y2))
            crop_tensor = self.transform(crop)
            crops.append(crop_tensor)
        
        # Process in batches
        all_features = []
        
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device)
            
            # Extract features
            features = self.model(batch_tensor)
            all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)


def extract_clip_features(
    extractor: FeatureExtractor,
    loader: VolleyballDatasetLoader,
    video_id: str,
    frame_id: int,
    annotation: VolleyballAnnotation,
    dataset_root: Path
) -> Optional[Dict]:
    """
    Extract features for a single clip.
    
    Args:
        extractor: FeatureExtractor instance
        loader: VolleyballDatasetLoader instance
        video_id: Video ID
        frame_id: Frame ID
        annotation: Annotation for this frame
        dataset_root: Root directory of dataset
        
    Returns:
        Dictionary with features and metadata, or None if failed
    """
    try:
        # Get frame sequence (temporal window)
        frame_paths = loader.get_frame_sequence(video_id, frame_id)
        
        if not frame_paths:
            logger.warning(f"No frames found for {video_id}/{frame_id}")
            return None
        
        # Extract features for each frame in sequence
        sequence_features = []
        
        for frame_path in frame_paths:
            # Load image
            image = Image.open(frame_path).convert('RGB')
            
            # Get bounding boxes for this frame's ID
            current_frame_id = int(frame_path.stem)
            
            # For simplicity, use the target frame's bboxes for all frames
            # (Proper implementation would load annotations for each frame)
            bboxes = annotation.get_bboxes()
            
            # Extract features
            frame_features = extractor.extract_from_boxes(image, bboxes)
            sequence_features.append(frame_features)
        
        # Stack into array (seq_len, num_players, feature_dim)
        features = np.stack(sequence_features, axis=0)
        
        return {
            'features': features,
            'activity': annotation.activity_class,
            'actions': annotation.get_actions(),
            'frame_id': frame_id,
            'num_frames': len(frame_paths),
            'num_players': annotation.num_players
        }
        
    except Exception as e:
        logger.error(f"Error extracting features for {video_id}/{frame_id}: {e}")
        return None


def extract_video_features(
    extractor: FeatureExtractor,
    loader: VolleyballDatasetLoader,
    video_id: str,
    output_dir: Path,
    dataset_root: Path
) -> int:
    """
    Extract features for all clips in a video.
    
    Args:
        extractor: FeatureExtractor instance
        loader: VolleyballDatasetLoader instance
        video_id: Video ID
        output_dir: Directory to save features
        dataset_root: Root directory of dataset
        
    Returns:
        Number of clips processed
    """
    # Load annotations for this video
    try:
        annotations = loader.load_video_annotations(video_id)
    except Exception as e:
        logger.error(f"Failed to load annotations for video {video_id}: {e}")
        return 0
    
    if not annotations:
        logger.warning(f"No annotations found for video {video_id}")
        return 0
    
    # Create output directory for this video
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    processed = 0
    
    for frame_id, annotation in tqdm(
        annotations.items(),
        desc=f"Video {video_id}",
        leave=False
    ):
        # Check if already processed
        output_file = video_output_dir / f"{frame_id}.npy"
        if output_file.exists():
            processed += 1
            continue
        
        # Extract features
        features_dict = extract_clip_features(
            extractor=extractor,
            loader=loader,
            video_id=video_id,
            frame_id=frame_id,
            annotation=annotation,
            dataset_root=dataset_root
        )
        
        if features_dict is not None:
            # Save features
            np.save(output_file, features_dict)
            processed += 1
    
    return processed


def main():
    """Main feature extraction pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Extract CNN features from volleyball dataset"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/volleyball_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='alexnet',
        choices=['alexnet', 'resnet50', 'vgg16'],
        help='CNN model to use'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/volleyball/features',
        help='Output directory for features'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to process'
    )
    
    parser.add_argument(
        '--videos',
        type=str,
        nargs='+',
        default=None,
        help='Specific video IDs to process'
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = Path(config['dataset']['root'])
    
    # Initialize feature extractor
    logger.info(f"Initializing feature extractor: {args.model}")
    extractor = FeatureExtractor(
        model_name=args.model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize dataset loader
    loader = VolleyballDatasetLoader(
        dataset_root=dataset_root,
        config=config['dataset']
    )
    
    # Determine which videos to process
    if args.videos:
        video_ids = args.videos
        logger.info(f"Processing specified videos: {video_ids}")
    elif args.split == 'all':
        video_ids = (
            config['dataset']['train_videos'] +
            config['dataset']['val_videos'] +
            config['dataset'].get('test_videos', [])
        )
        logger.info(f"Processing all videos: {len(video_ids)} videos")
    else:
        video_ids = config['dataset'][f'{args.split}_videos']
        logger.info(f"Processing {args.split} split: {len(video_ids)} videos")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Process each video
    logger.info(f"\nStarting feature extraction...")
    logger.info("=" * 70)
    
    total_clips = 0
    
    for i, video_id in enumerate(video_ids, 1):
        logger.info(f"\n[{i}/{len(video_ids)}] Processing video {video_id}")
        
        # Check if video exists
        video_dir = dataset_root / video_id
        if not video_dir.exists():
            logger.warning(f"Video directory not found: {video_dir}")
            continue
        
        # Extract features
        num_clips = extract_video_features(
            extractor=extractor,
            loader=loader,
            video_id=video_id,
            output_dir=output_dir,
            dataset_root=dataset_root
        )
        
        total_clips += num_clips
        logger.info(f"Processed {num_clips} clips from video {video_id}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total videos processed: {len(video_ids)}")
    logger.info(f"Total clips processed: {total_clips}")
    logger.info(f"Features saved to: {output_dir}")
    logger.info(f"Feature dimension: {extractor.feature_dim}")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()