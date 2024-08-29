import torch
import wandb

def evaluate_model(model, test_data, device='cuda'):
    model.eval()
    with torch.no_grad():
        input_ids = test_data['input_ids'].to(device)
        attention_mask = test_data['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze()

    predicted_y = predictions.cpu().numpy()

    wandb.log({"predictions": wandb.Histogram(predicted_y)})

    return predicted_y
