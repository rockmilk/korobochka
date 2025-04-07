import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=64, img_size=(224, 224)):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.indices = np.arange(len(images))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Изменение размера изображений
        batch_images_resized = tf.image.resize(batch_images, self.img_size).numpy()

        return batch_images_resized, batch_labels



train_generator = DataGenerator(x_train, y_train)
validation_generator = DataGenerator(x_test, y_test)


base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


base_model.trainable = False


model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=30,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))


test_loss, test_acc = model.evaluate(validation_generator)
print(f'\nТестовая точность: {test_acc:.4f}')


predictions = model.predict(validation_generator)



def plot_predictions(predictions, labels, num=10):
    plt.figure(figsize=(10, 5))
    for i in range(num):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(224, 224, 3))
        plt.title(f'Предсказание: {predictions[i].argmax()}, Истинная метка: {labels[i]}')
        plt.axis('off')
    plt.show()


plot_predictions(predictions, y_test)



def load_and_predict_image(filepath):
    img = Image.open(filepath).convert('L')  
    img = img.resize((28, 28))  
    img = np.array(img) / 255.0  # нормализация
    img = img.reshape((1, 28, 28, 1))  
    prediction = model.predict(img)
    return prediction.argmax()

def upload_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        prediction = load_and_predict_image(filepath)
        print(f'Предсказание для загруженного изображения: {prediction}')
        img = Image.open(filepath)
        plt.imshow(img, cmap='gray')
        plt.title(f'Предсказание: {prediction}')
        plt.axis('off')
        plt.show()


root = tk.Tk()
root.title("коробочка")


root.configure(bg='pink')


upload_button = tk.Button(root, text="Загрузить изображение", command=upload_image, bg='lightpink', fg='white', font=('Arial', 14))
upload_button.pack(pady=20)  

root.mainloop()
