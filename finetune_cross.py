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


val_generator = create_regular_generator(VAL_2_DIR, with_augment=False, shuffle=False)
callbacks_list = make_callbacks_list(MODEL_DIR / "finetune_cross_best.h5", val_generator)


# Load the pre-trained model from stage 1
model_path = MODEL_DIR / "finetune_orig_best.h5"
model = load_model_from_path(model_path)


# Recompile the model with a new learning rate
print(f"\nCompiling model with learning rate {LEARNING_RATE_2}")
compile_model(model, LEARNING_RATE_2)


# Cross-style training
for i, half in enumerate(["first_half", "second_half"]):
    
    train_dir = TRAIN_2_DIR / half
    train_generator = create_regular_generator(train_dir, with_augment=True, shuffle=True)

    print(f"\nStarting fine-tuning (Round {i})...")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    ## Save training artifacts
    print("\nSaving training artifacts...")
    save_training_artifacts(history, MODEL_DIR, f"finetune_cross_round_{i}")
    print("Artifacts saved successfully.")

