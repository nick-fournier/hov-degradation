"""Script for feed forward classification model"""

import tensorflow as tf


class FeedForwardClassifier:
    """Feed forward neural networks for classification.

    Attributes
    ----------
    # TODO
    """

    def __init__(self,
                 x,
                 y,
                 x_test,
                 y_test,
                 num_layers,
                 hidden_units,
                 activation_fn,
                 epochs,
                 batch_size,
                 learning_rate,
                 target_ckpt):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_ckpt = target_ckpt

        self.model = None

    def build_model(self):
        """
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=self.x.shape[1:]),
            tf.keras.layers.Dense(self.hidden_units,
                                  activation=self.activation_fn,
                                  input_shape=self.x.shape[1:],
                                  ),
            # tf.keras.layers.Dropout(rate=0.5),
            # tf.keras.layers.Dense(self.hidden_units,
            #                       activation=self.activation_fn),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(self.hidden_units,
                                  activation=self.activation_fn,
                                  kernel_regularizer=tf.keras.regularizers.l1(0.01)
                                  ),
            tf.keras.layers.Dropout(rate=0.5),

            tf.keras.layers.Dense(1,
                                  activation=tf.keras.activations.sigmoid,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01),)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=[tf.keras.metrics.Recall(),
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Accuracy()])
        return self.model

    def train(self):
        """
        """
        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.target_ckpt,
                                                         save_weights_only=True,
                                                         verbose=1)
        history = self.model.fit(x=self.x,
                                 y=self.y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 # callbacks=cp_callback,
                                 validation_data=(self.x_test, self.y_test)
                                 )
        return history

    def predict(self, xs):
        return self.model.predict(xs)

    def load_weights(self, path):
        return self.model.load_weights(path)

    def evaluate(self, xs, ys):
        return self.model.evaluate(xs, ys)