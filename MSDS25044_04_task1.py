# Complete Transformer QA System - All in One

import os
import re
import json
import argparse
import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu


# ============================================
# PART 1: TRANSFORMER MODEL (from scratch)
# ============================================

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attention_output = self.attention(query, key, value, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerForQA(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, 
                 d_ff=1024, max_seq_len=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def _generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        causal_mask = self._generate_causal_mask(seq_len).to(input_ids.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        padding_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        combined_mask = causal_mask & padding_mask
        
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, combined_mask)
        
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return logits, loss
    
    def generate(self, input_ids, tokenizer, max_new_tokens=100, temperature=1.0, do_sample=True):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.forward(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generated_ids = input_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text


# ============================================
# PART 2: DATA UTILS
# ============================================

class StackOverflowDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"<Usama_MSDS25044> {row['question']} </Usama_MSDS25044> {row['answer']}"
        
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'```.*?```', ' [CODE] ', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', ' [CODE] ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\:]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data(data_path, max_samples=15000):
    print("Loading data...")
    questions_df = pd.read_csv(f"{data_path}/Questions.csv", encoding='latin1')
    answers_df = pd.read_csv(f"{data_path}/Answers.csv", encoding='latin1')
    print(f"Loaded {len(questions_df)} questions and {len(answers_df)} answers")
    
    if 'AcceptedAnswerId' in questions_df.columns and 'Id' in answers_df.columns:
        merged_df = questions_df.merge(
            answers_df[['Id', 'Body']],
            left_on='AcceptedAnswerId',
            right_on='Id',
            how='inner'
        )
        merged_df = merged_df.rename(columns={
            'Title': 'title', 'Body_x': 'question_body', 'Body_y': 'answer'
        })
    else:
        merged_df = questions_df.merge(
            answers_df, left_on='Id', right_on='ParentId', how='inner'
        )
        merged_df = merged_df.rename(columns={
            'Title': 'title', 'Body_x': 'question_body', 'Body_y': 'answer'
        })
    
    print("Preprocessing text...")
    questions = []
    for _, row in tqdm(merged_df.iterrows(), total=min(len(merged_df), max_samples)):
        title = preprocess_text(row.get('title', ''))
        body = preprocess_text(row.get('question_body', ''))
        question = f"{title} {body}".strip()
        answer = preprocess_text(row.get('answer', ''))
        if len(question) > 10 and len(answer) > 10:
            questions.append({'question': question, 'answer': answer})
    
    data_df = pd.DataFrame(questions)
    if len(data_df) > max_samples:
        data_df = data_df.sample(n=max_samples, random_state=42)
    print(f"Preprocessed {len(data_df)} samples")
    return data_df


def create_data_loaders(data_df, tokenizer_name="openai-community/openai-gpt", 
                        batch_size=16, max_seq_len=256, val_split=0.15, test_split=0.15):
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_tokens = {'additional_special_tokens': ['<Usama_MSDS25044>', '</Usama_MSDS25044>']}
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_val_df, test_df = train_test_split(data_df, test_size=test_split, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_split/(1-test_split), random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    train_dataset = StackOverflowDataset(train_df, tokenizer, max_seq_len)
    val_dataset = StackOverflowDataset(val_df, tokenizer, max_seq_len)
    test_dataset = StackOverflowDataset(test_df, tokenizer, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, tokenizer


# ============================================
# PART 3: TRAINING FUNCTIONS
# ============================================

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


def compute_bleu(model, dataloader, tokenizer, device, max_samples=100):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples:
                break
            input_ids = batch['input_ids'].to(device)
            for j in range(input_ids.size(0)):
                question_ids = input_ids[j:j+1]
                generated_text = model.generate(question_ids, tokenizer, max_new_tokens=50, do_sample=False)
                if '</Usama_MSDS25044>' in generated_text:
                    answer_part = generated_text.split('</Usama_MSDS25044>')[-1].strip()
                else:
                    answer_part = generated_text
                ref_text = tokenizer.decode(batch['labels'][j].tolist(), skip_special_tokens=True)
                if '</Usama_MSDS25044>' in ref_text:
                    ref_answer = ref_text.split('</Usama_MSDS25044>')[-1].strip()
                else:
                    ref_answer = ref_text
                if len(answer_part) > 0 and len(ref_answer) > 0:
                    hypotheses.append(answer_part)
                    references.append([ref_answer])
    if len(hypotheses) > 0:
        bleu = corpus_bleu(hypotheses, references).score
    else:
        bleu = 0.0
    return bleu


def plot_results(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
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
    
    data_df = load_and_preprocess_data(args.data_path, max_samples=args.max_samples)
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        data_df, batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    
    vocab_size = len(tokenizer)
    model = TransformerForQA(
        vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_seq_len=args.max_seq_len,
        dropout=args.dropout, pad_idx=tokenizer.pad_token_id
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience)
    
    history = {'train_loss': [], 'val_loss': [], 'train_perplexity': [], 'val_perplexity': []}
    best_val_loss = float('inf')
    os.makedirs(args.save_path, exist_ok=True)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_ppl = compute_perplexity(train_loss)
        val_loss = validate_epoch(model, val_loader, device)
        val_ppl = compute_perplexity(val_loss)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_perplexity'].append(train_ppl)
        history['val_perplexity'].append(val_ppl)
        
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.save_path}/best_model.pt")
            print(f"✓ Saved best model")
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    with open(f"{args.save_path}/training_history.json", 'w') as f:
        json.dump(history, f)
    
    plot_results(history, args.save_path)
    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f}")
    
    # Compute BLEU on test set
    print("\nComputing BLEU score on test set...")
    bleu = compute_bleu(model, test_loader, tokenizer, device, max_samples=100)
    print(f"Test BLEU Score: {bleu:.2f}")
    
    return model, tokenizer, history


# ============================================
# PART 4: TESTING FUNCTIONS
# ============================================

def interactive_qa(model, tokenizer, device):
    print("\n" + "="*50)
    print("Interactive Q&A Mode")
    print("Enter your question (or 'quit' to exit)")
    print("="*50)
    
    while True:
        print("\n" + "-"*40)
        question = input("Your question: ").strip()
        if question.lower() == 'quit':
            break
        if len(question) == 0:
            continue
        
        processed_question = preprocess_text(question)
        text = f"<Usama_MSDS25044> {processed_question} </Usama_MSDS25044>"
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
        
        print("Generating answer...")
        answer = model.generate(input_ids, tokenizer, max_new_tokens=100, temperature=0.7, do_sample=True)
        
        if '</Usama_MSDS25044>' in answer:
            answer = answer.split('</Usama_MSDS25044>')[-1].strip()
        
        print(f"\nAnswer: {answer}")
        print("-"*40)


def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
    special_tokens = {'additional_special_tokens': ['<Usama_MSDS25044>', '</Usama_MSDS25044>']}
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    vocab_size = len(tokenizer)
    model = TransformerForQA(
        vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_seq_len=args.max_seq_len,
        dropout=args.dropout, pad_idx=tokenizer.pad_token_id
    ).to(device)
    
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    print(f"✓ Loaded weights from {args.weights_path}")
    
    if args.mode == 'test':
        print("\nLoading test data...")
        data_df = load_and_preprocess_data(args.data_path, max_samples=args.max_samples)
        _, _, test_loader, _ = create_data_loaders(data_df, batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        
        bleu = compute_bleu(model, test_loader, tokenizer, device, max_samples=args.max_test_samples)
        print("\n" + "="*50)
        print(f"Test BLEU Score: {bleu:.2f}")
        print("="*50)
    elif args.mode == 'interactive':
        interactive_qa(model, tokenizer, device)


# ============================================
# PART 5: MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Transformer QA System')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--save_path', type=str, default='./weights')
    parser.add_argument('--weights_path', type=str, default='./weights/best_model.pt')
    
    # Data parameters
    parser.add_argument('--max_samples', type=int, default=15000)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Test parameters
    parser.add_argument('--max_test_samples', type=int, default=100)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Transformer QA System - Usama Umer (MSDS25044)")
    print("="*60)
    
    if args.mode == 'train':
        train(args)
    else:
        test_model(args)


if __name__ == "__main__":
    main()