import tensorflow as tf

class Calculator(object):
    num1 = 0
    num2 = 0

    @property
    def num1(self)-> int: return self._num1

    @property
    def num2(self) -> int: return self._num2

    @num1.setter
    def num1(self, num1): self._num1 = num1

    @num2.setter
    def num2(self, num2): self._num2 = num2

class CalculatorService(object):

    @tf.function
    def add(self, x: Calculator): return tf.add(x.num1, x.num2)

    @tf.function
    def subtract(self, x: Calculator): return tf.subtract(x.num1, x.num2)

    @tf.function
    def multiply(self, x: Calculator): return tf.multiply(x.num1, x.num2)

    @tf.function
    def divide(self, x: Calculator): return tf.divide(x.num1, x.num2)

calculator_menu = ["Exit", #0
                "+", #1
                "-", #2
                "*", #3
                "/", #4
             ]
calculator_lambda = {
    "1" : lambda x, y: print(f" {y.num1} + {y.num2} = {x.add(y)}"),
    "2" : lambda x, y: print(f" {y.num1} + {y.num2} = {x.subtract(y)}"),
    "3" : lambda x, y: print(f" {y.num1} + {y.num2} = {x.multiply(y)}"),
    "4" : lambda x, y: print(f" {y.num1} + {y.num2} = {x.divide(y)}"),
}
if __name__ == '__main__':
    calculator = Calculator()
    calculator.num1 = 9
    calculator.num2 = 3
    service = CalculatorService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(calculator_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                calculator_lambda[menu](service, calculator)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")

