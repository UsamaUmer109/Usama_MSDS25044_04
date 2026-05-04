# Data Utilities for Stack Overflow Q&A Dataset


import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class StackOverflowDataset(Dataset):
    """Stack Overflow Q&A Dataset"""
    
    def __init__(self, data, tokenizer, max_seq_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Add custom tokens as required
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
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def preprocess_text(text):
    """Preprocess text: lowercase, remove HTML, remove special characters"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace code blocks with [CODE] token (optional but recommended)
    text = re.sub(r'```.*?```', ' [CODE] ', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', ' [CODE] ', text)
    
    # Remove special characters (keep alphanumeric, spaces, basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\:]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_preprocess_data(data_path, max_samples=15000):
    """Load CSV files and preprocess data"""
    print("Loading data...")
    questions_df = pd.read_csv(f"{data_path}/Questions.csv", encoding='latin1')
    answers_df = pd.read_csv(f"{data_path}/Answers.csv", encoding='latin1')
    
    print(f"Loaded {len(questions_df)} questions and {len(answers_df)} answers")
    
    # Merge questions with accepted answers
    if 'AcceptedAnswerId' in questions_df.columns and 'Id' in answers_df.columns:
        merged_df = questions_df.merge(
            answers_df[['Id', 'Body']],
            left_on='AcceptedAnswerId',
            right_on='Id',
            how='inner'
        )
        merged_df = merged_df.rename(columns={
            'Title': 'title',
            'Body_x': 'question_body',
            'Body_y': 'answer'
        })
    else:
        # Alternative merging strategy
        merged_df = questions_df.merge(
            answers_df,
            left_on='Id',
            right_on='ParentId',
            how='inner'
        )
        merged_df = merged_df.rename(columns={
            'Title': 'title',
            'Body_x': 'question_body',
            'Body_y': 'answer'
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
    """Create train, validation, and test data loaders"""
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add custom tokens as required
    special_tokens = {
        'additional_special_tokens': ['<Usama_MSDS25044>', '</Usama_MSDS25044>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Split data into train/val/test - NO DATA LEAKAGE
    train_val_df, test_df = train_test_split(
        data_df, test_size=test_split, random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_split/(1-test_split), random_state=42
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = StackOverflowDataset(train_df, tokenizer, max_seq_len)
    val_dataset = StackOverflowDataset(val_df, tokenizer, max_seq_len)
    test_dataset = StackOverflowDataset(test_df, tokenizer, max_seq_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, tokenizer