import tensorflow
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

seq_model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Input(shape=(28, 28, 1)),
    tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPooling2D(),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPooling2D(),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPooling2D(),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.GlobalAvgPool2D(),
    tensorflow.keras.layers.Dense(128, activation='relu'),
    tensorflow.keras.layers.Dense(10, activation='softmax')
])


# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
# y_test = tensorflow.keras.utils.to_categorical(y_test, 10)


def display_example(examples, labels):
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(examples[i], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(labels[i]))
    plt.show()


class MyCustomModel(tensorflow.keras.Model):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        self.conv1 = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = tensorflow.keras.layers.MaxPooling2D()
        self.bn1 = tensorflow.keras.layers.BatchNormalization()
        self.conv2 = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tensorflow.keras.layers.MaxPooling2D()
        self.bn2 = tensorflow.keras.layers.BatchNormalization()
        self.conv3 = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = tensorflow.keras.layers.MaxPooling2D()
        self.bn3 = tensorflow.keras.layers.BatchNormalization()
        self.global_pool = tensorflow.keras.layers.GlobalAvgPool2D()
        self.dense1 = tensorflow.keras.layers.Dense(128, activation='relu')
        self.dense2 = tensorflow.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.dense2(x)


model = MyCustomModel()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3,
          batch_size=64, validation_data=(x_test, y_test), validation_split=0.1)
model.evaluate(x_test, y_test, batch_size=64)
display_example(x_test, y_test)
