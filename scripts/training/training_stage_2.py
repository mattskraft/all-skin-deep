#!/usr/bin/env python
"""
Cross-style fine-tuning script for skin lesion classification.

This script performs cross-training between two halves of the dataset, where each half
contains a mix of original and style-transferred images. It loads a pre-trained model
from the first fine-tuning stage and trains it on each half, validating on the other.
This approach helps the model become more robust to different image styles.
"""

import argparse
from pathlib import Path
import sys
sys.path.append('..')
import config as cfg
import training_utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Cross-style fine-tuning of the pre-trained model from stage 1")
    
    # Load defaults from config module
    parser.add_argument("--data_dir", type=str, default=cfg.TRAIN_2_DIR,
                        help=f"Data directory (default: {cfg.TRAIN_2_DIR})")
    parser.add_argument("--model_dir", type=str, default=cfg.MODEL_DIR,
                        help=f"Model directory (default: {cfg.MODEL_DIR})")
    parser.add_argument("--learning_rate", type=float, default=cfg.LEARNING_RATE_2,
                        help=f"Learning rate (default: {cfg.LEARNING_RATE_2})")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help=f"Number of epochs (default: {cfg.NUM_EPOCHS})")
    parser.add_argument("--base_model", type=str, default="model_stage1_best.h5",
                        help="Base model to load (default: model_stage1_best.h5)")
    
    return parser.parse_args()

def main(args):
    # Convert string paths to Path objects
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    
    # Load the best model from the first fine-tuning stage
    model_path = model_dir / args.base_model
    print(f"Loading pre-trained model from {model_path}")
    model = utils.load_model_from_path(model_path)
    
    # Recompile the model with a lower learning rate for fine-tuning
    print(f"\nCompiling model with learning rate {args.learning_rate}")
    utils.compile_model(model, args.learning_rate)
    
    # Cross-style training on two different dataset halves
    for idx in range(2):
        
        # Train on one half, validate on the other
        if idx == 0:
            train_dir = data_dir / "first_half"
            val_dir = data_dir / "second_half"
        else:
            train_dir = data_dir / "second_half"
            val_dir = data_dir / "first_half"
        
        print(f"\nRound {idx+1}: Training on {train_dir}, validating on {val_dir}")
        print("-" * 50)
        
        # Create data generators for this round
        train_generator = utils.create_regular_generator(train_dir, with_augment=True, shuffle=True)
        val_generator = utils.create_regular_generator(val_dir, with_augment=False, shuffle=False)
        
        # Create fresh callbacks for this round with a unique best model path
        model_save_path = model_dir / f"model_stage2_round{idx+1}_best.h5"
        callbacks_list = utils.make_callbacks_list(model_save_path, val_generator)
        
        print(f"\nStarting fine-tuning (Round {idx+1} of 2)...")
        print(f"Training on {len(train_generator.filenames)} images")
        print(f"Validating on {len(val_generator.filenames)} images")
        print(f"Training for {args.epochs} epochs")
        
        # Train the model for this round
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator),
            epochs=args.epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save training artifacts (history and optional metrics)
        print(f"\nSaving training artifacts for round {idx+1}...")
        history_save_path = model_dir / f"training_history_stage2_round{idx+1}.csv"
        utils.save_training_artifacts(history, history_save_path)
        print(f"Best model saved to: {model_save_path}")
        print("Training artifacts saved successfully.")
    
    print("\nCross-style fine-tuning completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)