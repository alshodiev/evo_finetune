import torch
from torch.cuda.amp import autocast, GradScaler
import wandb
from torch.nn import MultiheadAttention as MHA


def train_model(model, optimizer, train_sequences, train_percentages, num_epochs=1, batch_size=2, device='cuda'):
    scaler = GradScaler()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i+batch_size]
            batch_labels = train_percentages[i:i+batch_size]

            # No need to convert batch_sequences and batch_labels to tensors again as they are already tensors
            input_ids = batch_sequences.to(device)
            labels = batch_labels.to(device)

            with autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        average_loss = epoch_loss / (len(train_sequences) / batch_size)
        wandb.log({'epoch': epoch + 1, 'loss': average_loss})
        print(f'Epoch: {epoch+1}, Loss: {average_loss}')
