from model_utils import (
    create_regular_generator,
    make_callbacks_list,
    save_training_artifacts,
    load_model_from_path,
    compile_model
)
from config import (
    TRAIN_2_DIR,
    VAL_2_DIR,
    MODEL_DIR,
    NUM_EPOCHS,
    LEARNING_RATE_2
)

# Load the best model from the first fine-tuning stage
# This model was saved by the ModelCheckpoint callback based on best validation F1-macro
model_path = MODEL_DIR / "finetune_orig_best.h5"
print(f"Loading pre-trained model from {model_path}")
model = load_model_from_path(model_path)

# Recompile the model with a lower learning rate for fine-tuning
# This preserves the model architecture and weights but updates the optimizer
print(f"\nCompiling model with learning rate {LEARNING_RATE_2}")
compile_model(model, LEARNING_RATE_2)

# Cross-style training on two different dataset halves
# Each half contains a mix of original and style-transferred images
for i, half in enumerate(["first_half", "second_half"]):
    
    # Each half has its own training and validation directories
    # This allows for cross-validation between the two halves
    train_dir = TRAIN_2_DIR / half
    val_dir = VAL_2_DIR / half
    
    print(f"\nRound {i+1}: Training on {train_dir}, validating on {val_dir}")
    
    # Create data generators for this round
    # Training data uses augmentation and shuffling for robustness
    train_generator = create_regular_generator(train_dir, with_augment=True, shuffle=True)
    # Validation data has no augmentation or shuffling for consistent evaluation
    val_generator = create_regular_generator(val_dir, with_augment=False, shuffle=False)
    
    # Create fresh callbacks for this round with a unique best model path
    # The ModelCheckpoint in these callbacks will save the best model based on validation F1-macro
    model_save_path = MODEL_DIR / f"finetune_cross_{i+1}_best.h5"
    callbacks_list = make_callbacks_list(model_save_path, val_generator)

    print(f"\nStarting fine-tuning (Round {i+1} of 2)...")
    print(f"Training on {len(train_generator.filenames)} images")
    print(f"Validating on {len(val_generator.filenames)} images")

    # Train the model for this round
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save training artifacts (history and optional metrics)
    # Note: The best model is already saved by the ModelCheckpoint callback
    print(f"\nSaving training artifacts for round {i+1}...")
    save_training_artifacts(history, MODEL_DIR, f"finetune_cross_round_{i+1}")
    print("Training artifacts saved successfully.")

print("\nCross-style fine-tuning completed.")
print(f"Best models saved to:")
print(f"  - {MODEL_DIR}/finetune_cross_1_best.h5")
print(f"  - {MODEL_DIR}/finetune_cross_2_best.h5")
