import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):  # LSTM
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)
        outputs, hidden = model(inputs, hidden)
        if isinstance(hidden, tuple):
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()
        loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    # Hyperparameters
    batch_size = 64
    hidden_size = 128
    n_layers = 2
    learning_rate = 0.001
    num_epochs = 1000
    early_stop_patience = 50
    dropout_prob = 0.3

    # dataset 불러오기
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', 'dataset', 'shakespeare_train.txt')
    dataset = Shakespeare(dataset_path)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # Initialize the model, criterion and optimizer
    model = CharLSTM(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers, dropout_prob=dropout_prob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    results = {'epoch': [], 'training_loss': [], 'validation_loss': []}
    best_val_loss = float('inf')
    patience_count = 0
    for epoch in range(num_epochs):
        trn_loss = train(model, trn_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')
        results['epoch'].append(epoch+1)
        results['training_loss'].append(trn_loss)
        results['validation_loss'].append(val_loss)

        # 조기 종료 검사
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            # 검증 손실이 개선될 때마다 모델 저장
            torch.save({
                'model_type': 'CharLSTM',  # 또는 'CharRNN' → 아무거나 상관없을듯(?)
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'output_size': model.output_size,
                'n_layers': model.n_layers,
                'dropout_prob': model.dropout_prob,
                'model_state_dict': model.state_dict(),
            }, 'best_model.pth')
        else:
             patience_count += 1
             if patience_count >= early_stop_patience:
                print(f'Validation loss did not improve for {early_stop_patience} epochs. Early stopping...')
                break


    #CSV로 저자ㅇ
    results_df = pd.DataFrame(results)
    results_df.to_csv('training_results.csv', index=False)

if __name__ == '__main__':
    main()
