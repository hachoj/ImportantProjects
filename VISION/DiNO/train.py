import torch
import torch.nn.functional as F
from data_loader import TinyImageNetDataLoader
from loss import compute_dino_loss
import json
import os
from datetime import datetime
import numpy as np
import math

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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
def train_dino(dino,
               data_loader,
               optimizer,
               device,
               config,
               save_dir='./checkpoints',
    ):
    """
    Main training loop for DINO
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

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    global_step = 0
    total_steps = config.num_epochs * len(data_loader.train_loader)
    warmup_steps = config.warmup_epochs * len(data_loader.train_loader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
     
    for epoch in range(config.num_epochs):
        print(f"Epoch: {epoch+1}/{config.num_epochs}")
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

            metrics['student_norm'].append(student_outputs.norm(dim=-1).mean().item())
            metrics['teacher_norm'].append(teacher_outputs.norm(dim=-1).mean().item())
            # Normalize student outputs
            student_outputs = F.normalize(student_outputs, dim=-1)

            # Normalize teacher outputs
            teacher_outputs = F.normalize(teacher_outputs, dim=-1)

            
            # Compute loss
            loss = compute_dino_loss(
                student_outputs, 
                teacher_outputs, 
                dino.center, 
                config.student_temp, 
                config.teacher_temp,
                global_step,
                total_steps
            )

            # Backpropagation with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dino.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()

            # Update teacher network
            dino.update_teacher(config.beta)

            # Update center
            with torch.no_grad():
                batch_center = teacher_outputs.mean(dim=(0, 1), keepdim=True).view(1, -1)
                dino.center = config.m * dino.center.view(1, -1) + (1 - config.m) * batch_center

            # Track metrics
            metrics['loss'].append(loss.item())
            metrics['steps'].append(global_step)
            metrics['center_norm'].append(dino.center.norm(dim=-1).mean().item())
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Print progress
            if global_step % config.print_every == 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"Step: {global_step}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"Student Norm (pre-norm): {metrics['student_norm'][-1]:.2f}, "
                    f"Teacher Norm (pre-norm): {metrics['teacher_norm'][-1]:.2f}, "
                    f"Center Norm: {metrics['center_norm'][-1]:.2f}"
                )
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches
        metrics['epoch_avg_losses'].append(avg_epoch_loss)
        
        # Save checkpoint if it's time or if it's the best model
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
        
        if (epoch + 1) % config.checkpoint_freq == 0 or is_best:
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