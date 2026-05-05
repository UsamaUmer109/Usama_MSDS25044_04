# Transformer Model for Question Answering


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention from 'Attention is All You Need'"""
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
    """Multi-Head Attention"""
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
        
        # Linear projections and reshape for multi-head
        query = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.attention(query, key, value, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attention_output)
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
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
    """Position-wise Feed-Forward Network"""
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
    """Single Transformer Block"""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerForQA(nn.Module):
    """Complete Transformer Model for Question Answering"""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, 
                 d_ff=1024, max_seq_len=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def _generate_causal_mask(self, size):
        """Generate causal mask for autoregressive decoding"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create masks
        causal_mask = self._generate_causal_mask(seq_len).to(input_ids.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        padding_mask = (input_ids != self.pad_idx).float().unsqueeze(1).unsqueeze(2)
        combined_mask = causal_mask & padding_mask
        
        # Embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, combined_mask)
        
        # Final layer
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return logits, loss
    
    def generate(self, input_ids, tokenizer, max_new_tokens=100, temperature=1.0, do_sample=True):
        """Generate answer from input question"""
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