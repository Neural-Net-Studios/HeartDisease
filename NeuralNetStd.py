from keras.models import Sequential
from keras import layers
class NeuralNetwork ():
    def __init__(self):
        self.model = Sequential()
    def inputlayer(self, next_layer, activation, input_neuros):
        self.model.add(layers.Dense(next_layer, activation=activation, input_dim=input_neuros))
    def hiddenlayer(self, dropout, next_layer, activation):
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Dense(next_layer, activation=activation))
    def outputlayer(self, next_layer, activation):
        self.model.add(layers.Dense(next_layer, activation=activation))
    def training(self, optimizer, loss, metrics, x_train, y_train, epochs):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        self.model.fit(x_train, y_train, epochs=epochs)
    def test(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print(score)
    def predict(self, data):
        print(self.model.predict(data))
    def predict_classes(self, data):
        print(self.model.predict_classes(data))
    def summary(self):
        print(self.model.summary())
