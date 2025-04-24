
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
train_generator, steps_per_epoch = create_weighted_generator(TRAIN_1_DIR)
val_generator = create_regular_generator(VAL_1_DIR)

callbacks_list = make_callbacks_list(MODEL_DIR / "finetune_orig_best.h5", val_generator)


## Build model
model = build_model(num_classes=len(CLASS_WEIGHTS))


## Compile model
print(f"\nCompiling model with learning rate {LEARNING_RATE_1}")
compile_model(model, LEARNING_RATE_1)


## Train model
print("\nStarting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=len(val_generator),
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)


## Save training artifacts
print("\nSaving training artifacts...")
save_training_artifacts(history, MODEL_DIR, "finetune_orig")
print("Artifacts saved successfully.")

print("\nTraining completed.")
