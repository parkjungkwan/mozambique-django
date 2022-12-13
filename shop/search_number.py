import pandas as pd
import tensorflow as tf

class Iris(object):
    def __init__(self):
        self.data = pd.read_csv(r'C:\Users\AIA\PycharmProjects\djangoProject\data\Iris.csv')

    def hook(self):
        self.spec()

    def spec(self):
        print(" --- 1.Shape ---")
        print(self.data.shape)
        print(" --- 2.Features ---")
        print(self.data.columns)
        print(" --- 3.Info ---")
        print(self.data.info())
        print(" --- 4.Case Top1 ---")
        print(self.data.head(1))
        print(" --- 5.Case Bottom1 ---")
        print(self.data.tail(3))
        print(" --- 6.Describe ---")
        print(self.data.describe())
        print(" --- 7.Describe All ---")
        print(self.data.describe(include='all'))


iris_menu = ["Exit", #0
                "hook"] #1
iris_lambda = {
    "1" : lambda x: x.hook(),
}
if __name__ == '__main__':
    iris = Iris()
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