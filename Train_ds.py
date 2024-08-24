import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# Function to load and preprocess images
def load_images(data_path, target_size=(192, 192)):
    images = []
    labels = []
    class_labels = sorted(os.listdir(data_path))
    
    for label, class_name in enumerate(class_labels):
        class_path = os.path.join(data_path, class_name)
        for file in glob.glob(class_path + "/*.jpg"):
            img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Define paths to your dataset
train_data_dir = r'Path\raw'  # Training data directory

# Load and preprocess images
X, y = load_images(train_data_dir)

# Normalize pixel values to range [0, 1]
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
# Check if label array is empty
if len(y) == 0:
    raise ValueError("No labels found. Make sure your labels are assigned correctly.")
else:
    # Convert labels to one-hot encoded vectors
    y = to_categorical(y)

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(192, 192, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks for early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('Path\model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid), callbacks=[early_stop, checkpoint])



print("Model training complete.")
