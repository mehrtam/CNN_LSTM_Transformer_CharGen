import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load Wikipedia Text (first 100K characters)
dataset = datasets.load_dataset("wikitext", "wikitext-103-v1", split="train")
wiki_text = " ".join(dataset["text"])[:10000]

# Character Vocabulary
chars = sorted(set(wiki_text))
char2idx = {char: idx for idx, char in enumerate(chars)}
idx2char = {idx: char for char, idx in char2idx.items()}
vocab_size = len(chars)

# Encode Text
def encode_text(text, seq_length=200):
    encoded = [char2idx[c] for c in text if c in char2idx]
    inputs, targets = [], []
    for i in range(len(encoded) - seq_length):
        inputs.append(encoded[i:i+seq_length])
        targets.append(encoded[i + seq_length])
    return torch.tensor(inputs), torch.tensor(targets)

# Dataset
class WikipediaDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Model
class NextCharPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, num_lstm_layers, num_transformer_layers, num_heads, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(embed_dim, lstm_units, num_lstm_layers, batch_first=True, bidirectional=True)
        transformer_layer = nn.TransformerEncoderLayer(d_model=lstm_units*2, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.fc = nn.Linear(lstm_units*2, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = self.fc(self.dropout(x[:, -1, :]))
        return x

# Evaluate on Validation Set
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_total += criterion(out, y).item()
    return loss_total / len(loader)

# Hyperparameter Tuning
optuna_logs = []
seq_length = 100
inputs, targets = encode_text(wiki_text, seq_length)
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.1, random_state=42)
train_loader = DataLoader(WikipediaDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(WikipediaDataset(X_val, y_val), batch_size=64)
# Global: Define outside the train_model
VALID_COMBINATIONS = [
    (h, e, u)
    for h in [2, 4, 6, 8]
    for e in [64, 128, 192, 256]
    for u in [128, 256, 384, 512]
    if (2 * u) % h == 0 and e % h == 0
]

def train_model(trial):
    global optuna_logs

    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 192, 256])
    lstm_units = trial.suggest_int("lstm_units", 128, 512, step=128)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 1, 2)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.0005])

    
    # Inside train_model
    num_heads, embed_dim, lstm_units = trial.suggest_categorical("head_embed_lstm", VALID_COMBINATIONS)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextCharPredictor(vocab_size, embed_dim, lstm_units, num_lstm_layers,
                              num_transformer_layers, num_heads, dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None
    patience = 3
    no_improve_epochs = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Trial {trial.number}, Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(best_model_state, f"best_model_trial_{trial.number}.pt")

    optuna_logs.append({
        "trial": trial.number,
        "embed_dim": embed_dim,
        "lstm_units": lstm_units,
        "num_lstm_layers": num_lstm_layers,
        "num_transformer_layers": num_transformer_layers,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "final_val_loss": best_val_loss
    })

    return best_val_loss


# All your imports and definitions here (model, dataset, functions)

def run_training():
    study = optuna.create_study(direction="minimize")
    study.optimize(train_model, n_trials=3)

    df = pd.DataFrame(optuna_logs)
    df.to_csv("optuna_results.csv", index=False)
    print(df)
    print("Best Trial:", study.best_params)




def generate_text(model, start_text, char2idx, idx2char, length=200, temperature=0.6):
    model.eval()
    input_seq = [char2idx[c] for c in start_text[-seq_length:] if c in char2idx]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)

    result = start_text

    for _ in range(length):
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output / temperature, dim=-1).squeeze()
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            result += idx2char[next_char_idx]

            input_tensor = torch.cat([
                input_tensor[:, 1:], torch.tensor([[next_char_idx]]).to(input_tensor.device)
            ], dim=1)

    return result

import os
import torch
import torch.nn.functional as F

def evaluate_next_char_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # Shape: (batch_size, vocab_size)
            preds = torch.argmax(logits, dim=-1)  # Most likely char index

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = 100.0 * correct / total
    print(f"✅ Next Character Prediction Accuracy: {accuracy:.2f}%")
    return accuracy

import torch
import torch.nn.functional as F

def evaluate_on_sentences(model, sentences, char2idx, idx2char, device):
    model.eval()
    total_chars = 0
    correct_preds = 0

    for sentence in sentences:
        sentence = sentence.strip()
        for i in range(1, len(sentence)):
            context = sentence[:i]
            true_next_char = sentence[i]

            # Encode context
            input_indices = [char2idx.get(c, char2idx[" "]) for c in context]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

            # Predict next char
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                pred_char = idx2char[pred_idx]

            if pred_char == true_next_char:
                correct_preds += 1
            total_chars += 1

    accuracy = 100.0 * correct_preds / total_chars
    print(f"📊 Sentence-based Next-Char Accuracy: {accuracy:.2f}%")
    return accuracy

def run_generation():
    # Load the best trial info from CSV
    df = pd.read_csv("optuna_results.csv")
    best_trial_row = df.loc[df["final_val_loss"].idxmin()]
    best_trial = int(best_trial_row["trial"])

    # Extract hyperparameters
    embed_dim = int(best_trial_row["embed_dim"])
    lstm_units = int(best_trial_row["lstm_units"])
    num_lstm_layers = int(best_trial_row["num_lstm_layers"])
    num_transformer_layers = int(best_trial_row["num_transformer_layers"])
    num_heads = int(best_trial_row["num_heads"])
    dropout_rate = float(best_trial_row["dropout_rate"])

    # Model path
    model_path = f"best_model_trial_{best_trial}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with best hyperparameters
    
    model = NextCharPredictor(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        lstm_units=lstm_units,
        num_lstm_layers=num_lstm_layers,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate prediction accuracy on validation set
    evaluate_next_char_accuracy(model, val_loader, device)

    sentences = [
        "AI is the future of technology.",
        "Transformers changed the NLP world.",
        "Deep learning models are powerful.",
        "PyTorch makes experimentation easy.",
        "Hello world!"
    ]

    # Load best model
    best_model_path = f"best_model_trial_{best_trial}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

   
    # Evaluate on sentence typing
    evaluate_on_sentences(model, sentences, char2idx, idx2char, device)


    # Generate text
    text = generate_text(
        model,
        start_text="AI is the future of th",
        char2idx=char2idx,
        idx2char=idx2char,
        length=100,
        temperature=0.3
    )
    print(f"\nLoaded best model from trial {best_trial} with val_loss={best_trial_row['final_val_loss']:.4f}")
    print("\nGenerated Text:\n", text)



if __name__ == "__main__":
    # Comment out what you don't want
    #run_training()
    run_generation()
