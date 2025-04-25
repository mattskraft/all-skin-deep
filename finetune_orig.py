from model_utils import (
    create_weighted_generator,
    create_regular_generator,
    build_model,
    make_callbacks_list,
    save_training_artifacts,
    compile_model
)
from config import (
    TRAIN_1_DIR,
    VAL_1_DIR,
    MODEL_DIR,
    LEARNING_RATE_1,
    CLASS_WEIGHTS,
    NUM_EPOCHS,
)


## Get data generators
# Create a weighted training generator to handle class imbalance
# Returns both the generator and the calculated steps_per_epoch
train_generator, steps_per_epoch = create_weighted_generator(TRAIN_1_DIR)
print(f"Created weighted training generator with {steps_per_epoch} steps per epoch")

# Create a regular validation generator (no weighting or augmentation)
val_generator = create_regular_generator(VAL_1_DIR)
print(f"Created validation generator with {len(val_generator.filenames)} images")

# Create callbacks including ModelCheckpoint to save the best model
# based on validation F1-macro score
callbacks_list = make_callbacks_list(MODEL_DIR / "finetune_orig_best.h5", val_generator)
print("Configured callbacks with best model checkpoint")


## Build model
# Create a new model with the specified number of output classes
print(f"\nBuilding model with {len(CLASS_WEIGHTS)} output classes")
model = build_model(num_classes=len(CLASS_WEIGHTS))


## Compile model
# Configure the model with loss function, optimizer and metrics
print(f"Compiling model with learning rate {LEARNING_RATE_1}")
compile_model(model, LEARNING_RATE_1)


## Train model
print("\nStarting model training...")
print(f"Training for {NUM_EPOCHS} epochs")
print(f"Training on weighted samples from {TRAIN_1_DIR}")
print(f"Validating on {VAL_1_DIR}")

# Fit the model, using the weighted generator and callbacks
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,  # Use the calculated steps based on weighted sampling
    validation_steps=len(val_generator),
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)


## Save training artifacts
# Save the training history and related metrics
# Note: The best model is already saved by the ModelCheckpoint callback
print("\nSaving training artifacts...")
save_training_artifacts(history, MODEL_DIR, "finetune_orig")
print("Artifacts saved successfully.")

print("\nTraining completed.")
print(f"Best model saved to: {MODEL_DIR}/finetune_orig_best.h5")
print(f"Training history saved to: {MODEL_DIR}/finetune_orig_history.csv")
