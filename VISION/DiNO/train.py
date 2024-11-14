import torch
from data_loader import TinyImageNetDataLoader
from loss import compute_dino_loss
import json
import os
from datetime import datetime
import numpy as np

def save_metrics(metrics, save_dir):
    """Save metrics to a JSON file"""
    # Convert numpy values to float for JSON serialization
    for key in metrics:
        metrics[key] = [float(val) for val in metrics[key]]
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(save_dir, f'metrics_{timestamp}.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_path}")
    return metrics_path

def save_checkpoint(model, optimizer, epoch, metrics, save_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if specified
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

# Create learning rate scheduler
def get_lr(epoch, warmup_epochs, base_lr, min_lr, num_epochs):
    if epoch < warmup_epochs:
        # Linear warmup
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
def train_dino(dino,
               data_loader,
               optimizer,
               device,
               num_epochs,
               base_lr,
               min_lr,
               warmup_epochs,
               save_dir='./checkpoints',
               checkpoint_freq=5,  # Save checkpoint every N epochs
               tps=0.9,
               tpt=0.04,
               beta=0.9,
               m=0.9):
    """
    Args:
        dino: DINO Module
        data_loader: DataLoader for training
        optimizer: Optimizer (e.g., SGD, AdamW)
        device: torch.device ('cuda' or 'cpu')
        num_epochs: Number of epochs
        base_lr: Base learning rate after warmup
        min_lr: Minimum learning rate
        warmup_epochs: Number of warmup epochs
        save_dir: Directory to save checkpoints and metrics
        checkpoint_freq: Frequency of saving checkpoints (in epochs)
        tps: Student temperature
        tpt: Teacher temperature
        beta: Teacher EMA decay
        m: Center moving average decay
    Returns:
        dict: Training metrics history
    """
    # Initialize metric tracking
    metrics = {
        'loss': [],
        'student_norm': [],
        'teacher_norm': [],
        'center_norm': [],
        'steps': [],
        'epoch_avg_losses': []  # Track average loss per epoch
    }
    
    global_step = 0
    total_steps = num_epochs * len(data_loader.train_loader)
    print_every = 50  # Print metrics every 50 steps
    best_loss = float('inf')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Update learning rate
        current_lr = get_lr(epoch, warmup_epochs, base_lr, min_lr, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print(f"Epoch: {epoch+1}/{num_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for (global_views, local_views), _ in data_loader.train_loader:
            # Move to device
            global_views = global_views.to(device)
            local_views = local_views.to(device)
            
            # Reshape views for processing
            b = global_views.shape[0]
            global_views = global_views.reshape(-1, *global_views.shape[2:])
            local_views = local_views.reshape(-1, *local_views.shape[2:])
            
            # Forward pass through student
            all_views = torch.cat([global_views, local_views], dim=0)
            student_outputs = dino.student(all_views)
            student_outputs = student_outputs.reshape(b, 8, -1)
            
            # Forward pass through teacher
            with torch.no_grad():
                teacher_outputs = dino.teacher(global_views).reshape(b, 2, -1)
            
            # Compute loss
            loss = compute_dino_loss(
                student_outputs, 
                teacher_outputs, 
                dino.center, 
                tps, 
                tpt,
                global_step,
                total_steps
            )

            # Backpropagation with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dino.parameters(), max_norm=3.0)
            optimizer.step()

            # Update teacher network
            dino.update_teacher(beta)

            # Update center
            with torch.no_grad():
                batch_center = teacher_outputs.mean(dim=(0, 1), keepdim=True).view(1, -1)
                dino.center = m * dino.center.view(1, -1) + (1 - m) * batch_center

            # Track metrics
            metrics['loss'].append(loss.item())
            metrics['steps'].append(global_step)
            metrics['student_norm'].append(student_outputs.norm(dim=-1).mean().item())
            metrics['teacher_norm'].append(teacher_outputs.norm(dim=-1).mean().item())
            metrics['center_norm'].append(dino.center.norm(dim=-1).mean().item())
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Print progress
            if global_step % print_every == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"Step: {global_step}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"Student Norm: {metrics['student_norm'][-1]:.2f}, "
                    f"Teacher Norm: {metrics['teacher_norm'][-1]:.2f}, "
                    f"Center Norm: {metrics['center_norm'][-1]:.2f}"
                )
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches
        metrics['epoch_avg_losses'].append(avg_epoch_loss)
        
        # Save checkpoint if it's time or if it's the best model
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
        
        if (epoch + 1) % checkpoint_freq == 0 or is_best:
            save_checkpoint(dino, optimizer, epoch, metrics, save_dir, is_best)
        
        # Save metrics after each epoch
        save_metrics(metrics, save_dir)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Best Loss: {best_loss:.4f}")
        print("-" * 50)

    return metrics

# Example usage for loading checkpoints:
"""
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

# To load:
epoch, metrics = load_checkpoint('checkpoints/best_model.pt', model, optimizer)
"""
