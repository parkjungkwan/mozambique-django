import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

'''
Iris Species
Classify iris plants into three species in this classic dataset
'''
class IrisModel(object):
    def __init__(self):
        self.iris = datasets.load_iris()
        print(f'type {type(self.iris)}') # type <class 'sklearn.utils._bunch.Bunch'>
        self._X = self.iris.data
        self._Y = self.iris.target


    def hook(self):
        # self.spec()
        self.create_model()

    def spec(self):
        print(f'{self.iris.feature_names}')
        '''
        Shape (150, 6)
        ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']
        '''

    def create_model(self):
        X = self._X
        Y = self._Y
        enc = OneHotEncoder()
        Y_1hot = enc.fit_transform(Y.reshape(-1,1)).toarray()
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y_1hot, epochs=300, batch_size=10)
        print('Model Training is completed')

        file_name = './save/iris_model.h5'
        model.save(file_name)
        print(f'Model Saved in {file_name}')






iris_menu = ["Exit", #0
                "hook"] #1
iris_lambda = {
    "1" : lambda x: x.hook(),
}
if __name__ == '__main__':
    iris = IrisModel()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")