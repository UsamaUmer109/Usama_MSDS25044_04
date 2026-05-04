# Training Script for Transformer QA Model


import os
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import TransformerForQA
from data_utils import load_and_preprocess_data, create_data_loaders


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_perplexity(loss):
    return np.exp(loss)


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        _, loss = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            _, loss = model(input_ids, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def plot_results(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Perplexity plot
    axes[1].plot(history['train_perplexity'], label='Train Perplexity', marker='o')
    axes[1].plot(history['val_perplexity'], label='Val Perplexity', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training and Validation Perplexity')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('../graphs', exist_ok=True)
    plt.savefig(f"{save_path}/loss_perplexity_plots.png", dpi=300)
    plt.savefig("../graphs/loss_perplexity_plots.png", dpi=300)
    plt.show()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed(args.seed)
    
    # Load data
    data_df = load_and_preprocess_data(args.data_path, max_samples=args.max_samples)
    
    # Create loaders
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        data_df, batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    
    # Create model
    vocab_size = len(tokenizer)
    model = TransformerForQA(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pad_idx=tokenizer.pad_token_id
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (AdamW as required)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler (CosineAnnealingLR as required)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    early_stopping = EarlyStopping(patience=args.patience)
    
    history = {'train_loss': [], 'val_loss': [], 'train_perplexity': [], 'val_perplexity': []}
    best_val_loss = float('inf')
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_ppl = compute_perplexity(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        val_ppl = compute_perplexity(val_loss)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_perplexity'].append(train_ppl)
        history['val_perplexity'].append(val_ppl)
        
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        
        # Save best model (checkpointing as required)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.save_path}/best_model.pt")
            print(f"✓ Saved best model")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save history
    with open(f"{args.save_path}/training_history.json", 'w') as f:
        json.dump(history, f)
    
    # Plot graphs
    plot_results(history, args.save_path)
    
    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f}")
    
    return model, tokenizer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./weights')
    parser.add_argument('--max_samples', type=int, default=15000)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    train(args)