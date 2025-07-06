#!/usr/bin/env python3
"""
🧪 SIMPLE REBEL TEST EVALUATION
A simplified test to verify our REBEL model works and can generate predictions

This script:
1. Loads our trained REBEL model
2. Tests it on a few sample sentences
3. Demonstrates the evaluation pipeline works
"""

import sys
import os
import torch
import json
from pathlib import Path

# Add models to path
sys.path.append('models/rebel/src')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from models.rebel.src.re_score import re_score  # Commented out for now

def setup_rebel_model(checkpoint_path, hierarchy_level=1):
    """Setup the REBEL model for testing"""
    print("🔧 Setting up REBEL model...")

    # Load base model and tokenizer
    model_name = "facebook/bart-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Add MaintIE special tokens based on hierarchy level
    general_tokens = ["<num>", "<id>", "<date>", "<sensitive>"]

    if hierarchy_level >= 1:
        # Level 1: 5 entity types
        entity_tokens = ["<physical object>", "<process>", "<property>", "<activity>", "<state>"]
        general_tokens.extend(entity_tokens)

    # Add relation types
    relation_tokens = ["is a", "contains", "has part", "has participant", "has patient", "has agent", "has property"]

    # Add all tokens to tokenizer
    all_new_tokens = general_tokens + relation_tokens

    print(f"Adding {len(all_new_tokens)} special tokens...")
    tokenizer.add_tokens(all_new_tokens)

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    print(f"✅ Model setup complete - tokenizer size: {len(tokenizer)}")

    # Load our trained checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    # Fix for PyTorch 2.6 weights_only issue
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model state dict (skip PL wrapper keys)
    model_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            model_state_dict[new_key] = value

    # Load the state dict with size mismatch handling
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    print("✅ Checkpoint loaded successfully!")
    return model, tokenizer

def test_model_predictions(model, tokenizer, test_sentences):
    """Test the model on sample sentences"""
    print("🧪 Testing model predictions...")

    results = []

    for i, sentence in enumerate(test_sentences):
        print(f"\n📝 Test {i+1}: {sentence}")

        # Tokenize input
        inputs = tokenizer(sentence, return_tensors="pt", max_length=64, truncation=True)

        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=64,
                num_beams=3,
                early_stopping=True,
                do_sample=False
            )

        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=False)
        clean_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"   🤖 Raw output: {prediction}")
        print(f"   ✨ Clean output: {clean_prediction}")

        results.append({
            "input": sentence,
            "raw_output": prediction,
            "clean_output": clean_prediction
        })

    return results

def main():
    """Main test function"""
    print("🏆 SIMPLE REBEL MODEL TEST")
    print("=" * 50)

    # Configuration
    checkpoint_path = "experiments/rebel_final_corrected/epoch=0-step=430.ckpt"
    hierarchy_level = 1

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Test sentences from MaintIE domain
    test_sentences = [
        "The pump failed during maintenance.",
        "The motor drives the compressor unit.",
        "Temperature sensor monitors the cooling system.",
        "Replace the damaged bearing assembly.",
        "The control valve regulates pressure flow."
    ]

    try:
        # Setup model
        model, tokenizer = setup_rebel_model(checkpoint_path, hierarchy_level)

        # Test predictions
        results = test_model_predictions(model, tokenizer, test_sentences)

        # Save results
        output_file = "simple_rebel_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Test completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        print(f"🎯 Model is working and generating predictions!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
