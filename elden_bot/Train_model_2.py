import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.applications.inception_v3 import InceptionV3  # Example of using a pretrained model

# Define the actions and number of actions
actions = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
num_actions = len(actions)

# Number of runs and folder strings
num_runs = 32
folderStrings = [f"test_images{i}" for i in range(num_runs + 1)]

# Load data from folders
data = []
labels = []
batch = 100

for folderString in folderStrings:
    try:
        init = 0
        end = batch
        while True:
            ins = np.load(f"{folderString}/{init}_{end}.npy")
            images = np.load(f"{folderString}/img{init}_{end}.npy")
            data.append(images)
            labels.append(ins)
            init += batch
            end += batch
    except FileNotFoundError:
        pass

# Stack data and labels
data = np.vstack(data)
labels = np.vstack(labels)

# Expand dimensions and transpose for model input
data = np.expand_dims(data, axis=-1).transpose(0, 2, 1, 3)

# Frame stacking logic
data1 = np.concatenate((np.array([data[0]]), data[:-1]), axis=0)
data2 = np.concatenate((np.array([data[0]]), data[:-1]), axis=0)
data = np.concatenate((np.concatenate((data2, data1), axis=-1), data), axis=-1)

# Shuffle data and labels
data = tf.random.shuffle(data, seed=42)
labels = tf.random.shuffle(labels, seed=42)

# Split data into training and testing sets
params = {
    "learning_rate": 3e-4,
    "epochs": 1000,
    "split": 9 / 10,
    "batch_size": 128,
    "shuffle": 100 * num_runs,
    "base_model": "custom"
}

split = int(params["split"] * len(data))
training_data, testing_data = data[:split], data[split:]
training_labels, testing_labels = labels[:split], labels[split:]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

# Configure batch size and shuffle buffer size
BATCH_SIZE = params["batch_size"]
SHUFFLE_BUFFER_SIZE = params["shuffle"]
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Define model architecture
inputs = tf.keras.layers.Input(shape=(244, 138, 3))
init = tf.keras.initializers.VarianceScaling(scale=2)
x = tf.keras.layers.Conv2D(32, 8, strides=(4, 4), activation="relu", kernel_initializer=init)(inputs)
x = tf.keras.layers.Conv2D(64, 4, strides=(3, 3), activation="relu", kernel_initializer=init)(x)
x = tf.keras.layers.Conv2D(64, 3, strides=(1, 1), activation="relu", kernel_initializer=init)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer=init)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(num_actions, activation="sigmoid")(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)

# Print model summary
model.summary()

# Compile model with optimizer, loss, and metrics
opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
loss = tf.keras.losses.BinaryCrossentropy()
metrics = ["accuracy"]
model.compile(optimizer=opt, loss=loss, metrics=metrics)

# Initialize Weights & Biases for experiment tracking
wandb.init(project="supervised_runs", entity="elden_ring_ai", name="video_example1")

# Define callbacks for model training
checkpoint_filepath = "./video_test/checkpoint"
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=10, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_weights_only=True, monitor="val_loss", verbose=1),
    WandbCallback()
]

# Compute class weights and print them
weights = [1 / (np.sum(labels[:, i]) + 1) * labels.shape[0] for i in range(len(labels[0]))]
class_weight = {i: weights[i] for i in range(len(weights))}
print(class_weight)

# Train the model
model.fit(
    train_dataset,
    epochs=params["epochs"],
    validation_data=test_dataset,
    callbacks=callbacks,
    class_weight=class_weight
)
