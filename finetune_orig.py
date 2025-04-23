
from config import (
    create_weighted_generator,
    create_regular_generator,
    build_model,
    make_callbacks_list,
    save_training_artifacts
)

from config import (
    weighted_focal_loss,
    top_3_accuracy,
    F1MacroScore,
)

from config import (
    TRAIN_1_DIR,
    VAL_1_DIR,
    OVERSAMPLE_FACTOR,
    BATCH_SIZE,
    CLASS_MULTIPLIERS,
    BLOCKS_TO_UNFREEZE,
    LEARNING_RATE,
    CLASS_WEIGHTS,
    NUM_EPOCHS,
)
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy


## Get data generators
train_generator = create_weighted_generator(TRAIN_1_DIR)
total_original_images = sum(
    len([f for f in os.listdir(os.path.join(TRAIN_1_DIR, class_name)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for class_name in CLASS_MULTIPLIERS
)
steps_per_epoch = int(total_original_images / BATCH_SIZE * OVERSAMPLE_FACTOR)

val_generator = create_regular_generator(VAL_1_DIR)


## Build model
model = build_model()
# Print unfreezing information
print(f"\nUnfreezing blocks: {', '.join(BLOCKS_TO_UNFREEZE)}")
# Count trainable vs non-trainable parameters
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters: {trainable_params + non_trainable_params:,}")


## Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=weighted_focal_loss(gamma=2.0, class_weights=CLASS_WEIGHTS),
    metrics=[categorical_accuracy, top_3_accuracy, F1MacroScore(num_classes=7)]
)


## Train model
print("\nStarting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=len(val_generator),
    epochs=NUM_EPOCHS,
    callbacks=make_callbacks_list(),
    verbose=1
)


## Save model
print("\nSaving model...")
save_training_artifacts(model, history, model_dir, model_name)