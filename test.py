"""
Testing Script for Transformer QA Model
Author: Usama Umer
Roll Number: MSDS25044
"""

import argparse
import torch
import re
from transformers import AutoTokenizer
from model import TransformerForQA
from data_utils import preprocess_text, load_and_preprocess_data, create_data_loaders
from sacrebleu import corpus_bleu


def compute_bleu(model, dataloader, tokenizer, device, max_samples=100):
    """Compute BLEU score on test set"""
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
                
                # Generate answer using greedy decoding
                generated_text = model.generate(
                    question_ids, tokenizer, max_new_tokens=50, do_sample=False
                )
                
                # Extract answer part
                if '</Usama_MSDS25044>' in generated_text:
                    answer_part = generated_text.split('</Usama_MSDS25044>')[-1].strip()
                else:
                    answer_part = generated_text
                
                # Get reference answer
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


def interactive_qa(model, tokenizer, device):
    """Interactive Q&A mode"""
    print("\n" + "="*50)
    print("Interactive Q&A Mode")
    print("Enter your question (or 'quit' to exit)")
    print("="*50)
    
    while True:
        print("\n" + "-"*40)
        question = input("Your question: ").strip()
        
        if question.lower() == 'quit':
            print("Exiting...")
            break
        
        if len(question) == 0:
            print("Please enter a question.")
            continue
        
        # Preprocess question
        processed_question = preprocess_text(question)
        
        # Add custom tokens
        text = f"<Usama_MSDS25044> {processed_question} </Usama_MSDS25044>"
        
        # Tokenize
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
        
        print("Generating answer...")
        
        # Generate answer
        answer = model.generate(input_ids, tokenizer, max_new_tokens=100, temperature=0.7, do_sample=True)
        
        # Extract answer part
        if '</Usama_MSDS25044>' in answer:
            answer = answer.split('</Usama_MSDS25044>')[-1].strip()
        
        print(f"\nAnswer: {answer}")
        print("-"*40)


def test_model(args):
    """Main test function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
    special_tokens = {'additional_special_tokens': ['<Usama_MSDS25044>', '</Usama_MSDS25044>']}
    tokenizer.add_special_tokens(special_tokens)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
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
    
    # Load weights
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    print(f"✓ Loaded weights from {args.weights_path}")
    
    if args.mode == 'test':
        # Test mode - compute BLEU score
        print("\n" + "="*50)
        print("Testing Mode - Computing BLEU Score")
        print("="*50)
        
        print("Loading test data...")
        data_df = load_and_preprocess_data(args.data_path, max_samples=args.max_samples)
        _, _, test_loader, _ = create_data_loaders(
            data_df, 
            batch_size=args.batch_size, 
            max_seq_len=args.max_seq_len
        )
        
        print("Computing BLEU score...")
        bleu_score = compute_bleu(model, test_loader, tokenizer, device, max_samples=args.max_test_samples)
        
        print("\n" + "="*50)
        print(f"Test BLEU Score: {bleu_score:.2f}")
        print("="*50)
        
    elif args.mode == 'interactive':
        # Interactive mode
        interactive_qa(model, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Transformer QA Model')
    
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset folder')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'interactive'], 
                        help='Mode: test (BLEU score) or interactive (Q&A)')
    
    # Test arguments
    parser.add_argument('--max_samples', type=int, default=15000, help='Maximum samples for loading')
    parser.add_argument('--max_test_samples', type=int, default=100, help='Samples for BLEU calculation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length')
    
    # Model arguments (must match training)
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    test_model(args)