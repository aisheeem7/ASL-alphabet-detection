from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

img_size = (64, 64)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    "asl_alphabet_train",
    target_size=img_size,
    batch_size=32,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    "asl_alphabet_train",
    target_size=img_size,
    batch_size=32,
    subset='validation'
)

# CNN Model
model_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model_cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model_cnn.fit(train_gen, epochs=10, validation_data=val_gen)

#Evaluate
loss_cnn, acc_cnn = model_cnn.evaluate(val_gen)
print("CNN Accuracy:", acc_cnn)