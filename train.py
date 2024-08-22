import torch
from torch.cuda.amp import autocast, GradScaler

def train_model(model, optimizer, train_sequences, train_percentages, num_epochs=1, batch_size=2, accumulation_steps=4, device='cuda'):
    scaler = GradScaler()
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i+batch_size]
            batch_labels = train_percentages[i:i+batch_size]

            # Convert to tensors and move to the device
            input_ids = torch.tensor(batch_sequences, dtype=torch.long).to(device)
            labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(device)

            with autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i // batch_size + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
