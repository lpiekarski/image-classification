import tensorflow as tf


class Classifier:
    def __init__(self, img_width, img_height, num_classes):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(img_height, img_width, 3)),
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def name(self):
        return "CNN_1"

    def train(self, train_ds, val_ds, epochs):
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def validation(self, val_ds):
        ev = self.model.evaluate(val_ds)
        return ev[1]

