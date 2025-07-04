import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Dataset import VideoDataset
from model import LRCN
from tqdm import tqdm
import numpy as np

# Configuration
CONFIG = {
    'SEQUENCE_LENGTH': 16,
    'BATCH_SIZE': 4,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'CLASSES': ['Fighting', 'Shooting', 'Explosion', 'RoadAccidents', 'Normal'],
    'DATA_ROOT': os.path.abspath('frames'),
    'MODEL_SAVE_PATH': 'saved_models',
    'TRAIN_RATIO': 0.8  # 80% training, 20% testing
}

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Setup directories
    ensure_dir(CONFIG['MODEL_SAVE_PATH'])
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and split dataset
    full_dataset = VideoDataset(
        root_dir=CONFIG['DATA_ROOT'],
        classes=CONFIG['CLASSES'],
        sequence_length=CONFIG['SEQUENCE_LENGTH']
    )
    
    # Split dataset
    train_size = int(CONFIG['TRAIN_RATIO'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"\nDataset sizes:")
    print(f"- Training: {len(train_dataset)} samples")
    print(f"- Testing: {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=2
    )

    # Initialize model
    model = LRCN(len(CONFIG['CLASSES'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

    # Training variables
    best_acc = 0.0
    model_save_path = os.path.join(CONFIG['MODEL_SAVE_PATH'], 'best_model.pth')

    print("\nStarting Training...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["EPOCHS"]}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * train_correct / train_total:.2f}%"
            })

        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"- Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"- Train Accuracy: {train_acc:.2f}%")
        print(f"- Test Accuracy: {test_acc:.2f}%")

        # Save model checkpoints
        torch.save(model.state_dict(), os.path.join(CONFIG['MODEL_SAVE_PATH'], f'epoch_{epoch+1}.pth'))
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with test accuracy: {best_acc:.2f}%")

    # Final save
    final_model_path = os.path.join(CONFIG['MODEL_SAVE_PATH'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete!")
    print(f"Final model saved to {final_model_path}")
    print(f"Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()