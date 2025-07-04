import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, root_dir, classes, sequence_length=16, transform=None):
        self.classes = classes
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._validate_dataset(root_dir)
    
    def _validate_dataset(self, root_dir):
        print("\nValidating dataset structure...")
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            
            video_folders = [f for f in os.listdir(class_dir) 
                           if os.path.isdir(os.path.join(class_dir, f))]
            
            print(f"\nProcessing {class_name} ({len(video_folders)} videos):")
            for video_folder in tqdm(video_folders):
                video_path = os.path.join(class_dir, video_folder)
                frames = sorted([f for f in os.listdir(video_path) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                if len(frames) >= self.sequence_length:
                    self.samples.append((video_path, class_idx, len(frames)))
                else:
                    print(f" - Skipping {video_folder} (only {len(frames)} frames)")
        
        if len(self.samples) == 0:
            raise ValueError("No valid video sequences found. Check:"
                          "\n1. Frame extraction completed successfully"
                          "\n2. Sequence length is appropriate for your videos"
                          "\n3. Directory structure is correct")
        
        print(f"\nDataset initialized with {len(self.samples)} valid sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, class_idx, total_frames = self.samples[idx]
        frames = sorted([f for f in os.listdir(video_path) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Handle variable-length sequences
        if total_frames >= self.sequence_length:
            start_idx = np.random.randint(0, total_frames - self.sequence_length)
            selected_frames = frames[start_idx:start_idx + self.sequence_length]
        else:
            # Repeat frames if sequence is too short
            repeat_factor = (self.sequence_length // total_frames) + 1
            selected_frames = (frames * repeat_factor)[:self.sequence_length]
        
        sequence = []
        for frame_name in selected_frames:
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to read frame: {frame_path}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            sequence.append(frame)
        
        return torch.stack(sequence), class_idx