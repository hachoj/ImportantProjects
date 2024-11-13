import torch
from data_loader import TinyImageNetDataLoader

def train_dino(dino,
               data_loader,
               optimizer,
               device,
               num_epochs,
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
        tps: Student temperature
        tpt: Teacher temperature
        beta: Teacher EMA decay
        m: Center moving average decay
    """
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for batch in data_loader.train_loader:
            views, _ = batch  # We can ignore labels for DiNO
            # Unpack views
            global_views = views[:2]  # First two are global views
            local_views = views[2:]   # Remaining are local views

            # Move views to device
            global_views = [view.to(device) for view in global_views]
            local_views = [view.to(device) for view in local_views]
            student_inputs = global_views + local_views

            # Forward pass through student
            student_outputs = [dino.student(view) for view in student_inputs]

            # Forward pass through teacher
            with torch.no_grad():
                teacher_outputs = [dino.teacher(view) for view in global_views]

            # Compute distillation loss
            loss = dino.distillation_loss(student_outputs, teacher_outputs, dino.center, tps, tpt)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher network
            dino.update_teacher(beta)

            # Update center
            with torch.no_grad():
                batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
                dino.center = m * dino.center + (1 - m) * batch_center
