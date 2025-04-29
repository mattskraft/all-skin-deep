#!/usr/bin/env python
"""
Fine-tunes a deep learning model on the original dataset.

This script loads a pre-trained model, creates weighted data generators to handle class
imbalance, and fine-tunes the model on the specified dataset. It saves the best model
based on validation F1-score and training history artifacts.
"""

import argparse
from pathlib import Path
import model_utils as utils
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model with original dataset")
    
    # Load defaults using the config module alias
    parser.add_argument("--train_dir", type=str, default=cfg.TRAIN_1_DIR,
                        help=f"Training directory (default: {cfg.TRAIN_1_DIR})")
    parser.add_argument("--val_dir", type=str, default=cfg.VAL_1_DIR,
                        help=f"Validation directory (default: {cfg.VAL_1_DIR})")
    parser.add_argument("--model_dir", type=str, default=cfg.MODEL_DIR,
                        help=f"Model directory (default: {cfg.MODEL_DIR})")
    parser.add_argument("--learning_rate", type=float, default=cfg.LEARNING_RATE_1,
                        help=f"Learning rate (default: {cfg.LEARNING_RATE_1})")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help=f"Number of epochs (default: {cfg.NUM_EPOCHS})")
    parser.add_argument("--model_name", type=str, default="finetune_orig",
                        help="Base name for saving model and artifacts (default: finetune_orig)")
    
    return parser.parse_args()

def main(args):
    # Convert string paths to Path objects
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    model_dir = Path(args.model_dir)
    model_name = args.model_name
    
    train_generator, steps_per_epoch = utils.create_weighted_generator(train_dir)
    print(f"Created weighted training generator with {steps_per_epoch} steps per epoch")

    val_generator = utils.create_regular_generator(val_dir)
    print(f"Created validation generator with {len(val_generator.filenames)} images")

    best_model_path = model_dir / f"{model_name}_best.h5"
    callbacks_list = utils.make_callbacks_list(best_model_path, val_generator)
    print(f"Configured callbacks with best model checkpoint at {best_model_path}")

    print(f"\nBuilding model with {len(cfg.CLASS_WEIGHTS)} output classes")
    model = utils.build_model(num_classes=len(cfg.CLASS_WEIGHTS))

    print(f"Compiling model with learning rate {args.learning_rate}")
    utils.compile_model(model, args.learning_rate)

    print("\nStarting model training...")
    print(f"Training for {args.epochs} epochs")
    print(f"Training on weighted samples from {train_dir}")
    print(f"Validating on {val_dir}")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=len(val_generator),
        epochs=args.epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    print("\nSaving training artifacts...")
    utils.save_training_artifacts(history, model_dir, model_name)
    print("Artifacts saved successfully.")

    print("\nTraining completed.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {model_dir}/{model_name}_history.csv")

if __name__ == "__main__":
    args = parse_args()
    main(args)