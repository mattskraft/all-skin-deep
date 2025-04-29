import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path
import pandas as pd
import os
import config as cfg


def top_3_accuracy(y_true, y_pred):
    """
    Calculate top-3 accuracy metric.
    
    Args:
        y_true: Ground truth labels in one-hot encoding
        y_pred: Predicted probabilities for each class
        
    Returns:
        Top-3 accuracy score
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def weighted_focal_loss(gamma=2.0, class_weights=cfg.CLASS_WEIGHTS):
    """
    Create a weighted focal loss function for handling class imbalance.
    
    Focal loss focuses more on hard examples by down-weighting easy examples.
    This implementation also supports class weights to handle imbalanced datasets.
    
    Args:
        gamma: Focusing parameter that controls how much to down-weight easy examples.
              Higher gamma values give more weight to hard, misclassified examples.
        class_weights: Dictionary mapping class indices to weights for each class.
                      Higher weights increase the importance of corresponding classes.
    
    Returns:
        Compiled weighted focal loss function that can be used as a Keras loss function
    """
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Initialize with ones if no class weights provided
        if class_weights is None:
            weights = tf.ones_like(y_true)
        else:
            # Convert class_weights dict to a tensor
            weights = tf.zeros_like(y_true)
            for class_idx, weight in class_weights.items():
                # Add weight for each class where y_true has 1
                class_mask = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), class_idx), tf.float32)
                class_mask = tf.expand_dims(class_mask, axis=-1)
                class_weight = tf.ones_like(y_true) * weight
                weights = weights + (class_mask * class_weight)

        # Calculate focal weight
        focal_weight = tf.pow(1.0 - y_pred, gamma)

        # Calculate the cross entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Apply both weights
        loss = weights * focal_weight * cross_entropy

        return K.sum(loss, axis=-1)

    return focal_loss


def build_model(num_classes, weights='imagenet'):
    """
    Build a MobileNetV2-based model with custom classification head.
    
    Creates a transfer learning model based on MobileNetV2, with selectively
    unfrozen blocks and a custom classification head for skin lesion classification.
    
    Args:
        num_classes: Number of output classes for the classifier
        weights: Pre-trained weights to use, 'imagenet' (default) or None
    
    Returns:
        Configured Keras Model ready to be compiled
    """
    base_model = MobileNetV2(
        weights=weights,  # 'imagenet' or None
        include_top=False,
        input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
    )

    # Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze selected blocks
    for layer in base_model.layers:
        if any(block in layer.name for block in cfg.BLOCKS_TO_UNFREEZE):
            layer.trainable = True

    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Print unfreezing information
    print(f"\nUnfreezing blocks: {', '.join(cfg.BLOCKS_TO_UNFREEZE)}")
    # Count trainable vs non-trainable parameters
    trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([K.count_params(w) for w in model.non_trainable_weights])
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate loss, optimizer and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for the Adam optimizer
        
    Returns:
        Compiled model ready for training
    """
    return model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=get_loss_function(class_weights=cfg.CLASS_WEIGHTS),
        metrics=[categorical_accuracy, top_3_accuracy, F1MacroScore(num_classes=len(cfg.CLASS_WEIGHTS))]
)


class F1MacroScore(tf.keras.metrics.Metric):
    """
    Custom F1 Macro Score metric for multi-class classification.
    
    Calculates F1 score for each class independently and averages them,
    giving equal weight to each class regardless of sample count.
    This is particularly useful for imbalanced datasets.
    """
    
    def __init__(self, num_classes, name='f1_macro', **kwargs):
        """
        Initialize the F1MacroScore metric.
        
        Args:
            num_classes: Total number of classes in the classification task
            name: Name of the metric
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Initialize state variables with correct parameter format
        self.true_positives = self.add_weight(
            name='true_positives', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(
            name='false_positives', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(
            name='false_negatives', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state based on new predictions.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Model predictions (probabilities)
            sample_weight: Optional sample weights
        """
        # Convert probabilities to class indices
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)

        # One-hot encode predictions and true values
        y_pred = tf.one_hot(y_pred, self.num_classes)
        y_true = tf.one_hot(y_true, self.num_classes)

        # Calculate TP, FP, FN for each class
        self.true_positives.assign_add(
            tf.reduce_sum(y_true * y_pred, axis=0))
        self.false_positives.assign_add(
            tf.reduce_sum((1 - y_true) * y_pred, axis=0))
        self.false_negatives.assign_add(
            tf.reduce_sum(y_true * (1 - y_pred), axis=0))

    def result(self):
        """
        Calculate F1 macro score based on accumulated statistics.
        
        Returns:
            F1 macro score averaged across all classes
        """
        # Calculate precision and recall for each class
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        # Calculate F1 score for each class
        f1_score = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

        # Return macro average (mean of all class F1 scores)
        return tf.reduce_mean(f1_score)

    def reset_state(self):
        """Reset all metric state variables to initial values."""
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))


class ClassMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor per-class metrics during training.
    
    Calculates and logs detailed metrics for each class at specified intervals,
    including precision, recall, F1 score, and confusion matrix.
    """
    
    def __init__(self, validation_data, class_names, log_interval=5):
        """
        Initialize the callback.
        
        Args:
            validation_data: Validation data generator
            class_names: List of class names in order matching model outputs
            log_interval: Epoch interval for logging class metrics
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_interval = log_interval
        self.class_metrics_history = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate and log class metrics at the end of specified epochs.
        
        Args:
            epoch: Current epoch index
            logs: Dictionary of logs from training
        """
        if (epoch + 1) % self.log_interval == 0 or epoch == 0:
            # Get predictions
            val_pred = np.argmax(self.model.predict(self.validation_data), axis=1)
            val_true = self.validation_data.classes

            # Calculate confusion matrix
            cm = confusion_matrix(val_true, val_pred)

            # Calculate per-class metrics
            report = classification_report(val_true, val_pred,
                                          target_names=self.class_names,
                                          output_dict=True)

            # Calculate and add F1 macro to logs
            f1_scores = [report[class_name]['f1-score'] for class_name in self.class_names]
            f1_macro = np.mean(f1_scores)
            if logs is not None:
                logs['val_f1_macro'] = f1_macro

            # Log info
            print(f"\nEpoch {epoch+1} - Class Metrics:")
            print(f"F1 Macro: {f1_macro:.4f}")
            for class_name in self.class_names:
                print(f"{class_name}: F1={report[class_name]['f1-score']:.4f}, "
                      f"Precision={report[class_name]['precision']:.4f}, "
                      f"Recall={report[class_name]['recall']:.4f}")

            # Store for later visualization
            self.class_metrics_history.append({
                'epoch': epoch + 1,
                'confusion_matrix': cm,
                'report': report,
                'f1_macro': f1_macro
            })


# Dictionary of custom objects for model saving/loading
CUSTOM_OBJECTS = {
    'top_3_accuracy': top_3_accuracy,
    'F1MacroScore': F1MacroScore
}


def get_loss_function(class_weights=None, gamma=2.0):
    """
    Return the weighted focal loss function with specified parameters.
    
    Args:
        class_weights: Dictionary of class weights for loss calculation
        gamma: Focusing parameter for focal loss
        
    Returns:
        Configured loss function
    """
    return weighted_focal_loss(gamma=gamma, class_weights=class_weights)


def load_model_from_path(model_path):
    """
    Load a model from the specified path, with proper handling of custom objects.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model ready to be compiled or used
    """
    print(f"Loading model from {model_path}")
    custom_objects = CUSTOM_OBJECTS.copy()
    custom_objects['focal_loss'] = get_loss_function(class_weights=cfg.CLASS_WEIGHTS)
    return load_model(model_path, custom_objects=custom_objects, compile=False)


def make_callbacks_list(model_save_path, val_generator):
    """
    Create a list of callbacks for model training.
    
    Includes:
    - ModelCheckpoint: Save the best model based on validation F1 macro
    - ReduceLROnPlateau: Reduce learning rate when metrics plateau
    - EarlyStopping: Stop training when metrics stop improving
    - ClassMetricsCallback: Calculate and log per-class metrics
    
    Args:
        model_save_path: Path to save the best model
        val_generator: Validation data generator
        
    Returns:
        List of callbacks for use with model.fit()
    """
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_f1_macro',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_f1_macro',
        factor=0.5,
        patience=4,
        verbose=1,
        mode='max',
        min_lr=0.00001
    )

    early_stopping = EarlyStopping(
        monitor='val_f1_macro',
        patience=8,
        verbose=1,
        restore_best_weights=True,
        mode='max'
    )

    class_metrics = ClassMetricsCallback(
        validation_data=val_generator,
        class_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
        log_interval=2
    )

    return [checkpoint, reduce_lr, early_stopping, class_metrics]


# Data augmentation pipeline for training
datagen_with_augment = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
)

# Data preprocessing pipeline for validation/testing (no augmentation)
datagen_only_preproc = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)


def create_regular_generator(dir, with_augment=False, shuffle=False, batch_size=cfg.BATCH_SIZE):
    """
    Create a data generator from directory with optional augmentation.
    
    Args:
        dir: Directory containing class subdirectories with images
        with_augment: Whether to apply data augmentation (default: False)
        shuffle: Whether to shuffle the data (default: False)
        batch_size: Batch size for the generator (default: from config)
        
    Returns:
        Keras DirectoryIterator for the specified directory
    """
    generator=datagen_with_augment if with_augment else datagen_only_preproc
    return generator.flow_from_directory(
        dir,
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        seed=42
    )


def create_weighted_generator(dir, with_augment=True, shuffle=True,
                              batch_size=cfg.BATCH_SIZE, class_multipliers=cfg.CLASS_MULTIPLIERS):
    """
    Creates a weighted generator with oversampling based on class weights.
    
    This generator applies class-based weighting to handle imbalanced datasets,
    giving more weight to underrepresented classes during batch selection.
    
    Args:
        dir: Directory containing class subdirectories with images
        with_augment: Whether to apply data augmentation (default: True)
        shuffle: Whether to shuffle the data (default: True)
        batch_size: Batch size for the generator (default: from config)
        class_multipliers: Dictionary mapping class names to sampling weights
        
    Returns:
        tuple: (generator, steps_per_epoch) - the data generator and calculated steps per epoch
    """
    # regular generator
    gen = create_regular_generator(dir, with_augment, shuffle, batch_size*cfg.OVERSAMPLE_FACTOR)

    # Calculate steps_per_epoch
    total_original_images = sum(
        len([f for f in os.listdir(os.path.join(dir, class_name)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for class_name in class_multipliers
    )
    steps_per_epoch = int(total_original_images / batch_size * cfg.OVERSAMPLE_FACTOR)

    # Get class indices
    class_indices = gen.class_indices
    class_weights_array = np.ones(len(class_indices))

    # If sampling weights provided, use them
    if class_multipliers:
        for class_name, weight in class_multipliers.items():
            # Get the numerical index corresponding to the class name
            idx = class_indices[class_name]
            class_weights_array[idx] = weight # Now use the numerical index

    # Create the generator function
    def weighted_generator():
        while True:
            # Get a large batch
            x_large, y_large = next(gen)

            # Get class indices for each sample
            class_idxs = np.argmax(y_large, axis=1)

            # Calculate selection probabilities based on weights
            probs = np.array([class_weights_array[idx] for idx in class_idxs])
            probs = probs / probs.sum()  # Normalize

            # Select indices based on weights
            selected_indices = np.random.choice(
                len(y_large), size=min(batch_size, len(y_large)),
                replace=False, p=probs
            )

            # Return weighted batch
            yield x_large[selected_indices], y_large[selected_indices]
    
    # Return both the generator and steps_per_epoch
    return tf.data.Dataset.from_generator(
        weighted_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(class_indices)), dtype=tf.float32)
        )
    ), steps_per_epoch


def save_training_artifacts(history, history_path):
    """
    Saves training history and optional class-level metrics to disk.
    
    Args:
        history: History object returned by model.fit()
        model_dir: Directory to save artifacts (can be Path or str)
        model_name: Base name for saved files (no extension)
    """
    history_dir = Path(history_path).parent

    try:
        history_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {history_dir}")
    except Exception as e:
        print(f"Error creating folder: {e}")

    # Save training history as CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")