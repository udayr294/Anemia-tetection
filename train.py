import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = "Dataset_sample"

# -------------------- DATA AUGMENTATION --------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

train_generator = datagen.flow_from_directory(   # to load training images from dataset for training
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # categorical for 4 classes
    subset='training'
)

val_generator = datagen.flow_from_directory(   # to load validation images from dataset for validation checking
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -------------------- MODEL --------------------
# pre_trained model(transfer learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Fine-tune last few layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------- TRAIN --------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -------------------- SAVE MODEL --------------------
model.save("anemia_model_4class.h5")

# -------------------- CHECK CLASS MAPPING --------------------
print("Class Mapping:", train_generator.class_indices)
print("Training Completed Successfully ✅")