import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


# исходные данные
input_values = numpy.array([[2, 28, 5, 64],
                            [2, 14, 6, 64],
                            [2, 14, 6, 64],
                            [2, 12, 10, 64],
                            [2, 14, 20, 128],
                            [4, 6, 18, 64],
                            [4, 14, 7, 128],
                            [4, 12, 8, 128],
                            [4, 12, 10, 128],
                            [6, 12, 12, 192],
                            [5, 16, 8, 160],
                            [8, 17, 14, 128],
                            [6, 12, 14, 192],
                            [6, 12, 14, 192],
                            [8, 7, 16, 128],
                            [12, 8, 15, 192],
                            [8, 12, 14, 256],
                            [8, 8, 14, 128],
                            [8, 8, 14, 256],
                            [12, 7, 16, 192]])

output_values = numpy.array([[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]]).T  # транспонируем веса
weights = numpy.random.random((4, 1))-1  # рандомим  20 весов от -1 до 1

for i in range(1):

    output_layer = sigmoid(numpy.dot(input_values, weights))  # считаем ответ
    mistake = output_values - output_layer  # сравниваем эталоны
    adj = numpy.dot(input_values.T, mistake * (output_layer * (1 - output_layer)))  # узнаём корректирующие веса
    weights += adj  # корректируем веса

print('Перованачальные ответы')

print('Полученные ответы')
print(output_layer)

my_value = numpy.array([[8, 4, 16, 256]])
output = sigmoid(numpy.dot(my_value, weights))

print("Ответ на входные параметры: ", my_value)
print(output)
