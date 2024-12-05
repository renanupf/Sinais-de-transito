import numpy as np  # Biblioteca para álgebra linear
import pandas as pd  # Biblioteca para manipulação de dados
import pickle  # Para manipulação de arquivos binários
import matplotlib.pyplot as plt  # Para plotagem de gráficos
from keras.utils import to_categorical  # Para transformar rótulos em formato categórico
from keras.models import Sequential  # Para criar modelos sequenciais
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Para gerar dados de imagem em tempo real
from keras.callbacks import LearningRateScheduler  # Para ajustar a taxa de aprendizado

# Carregando os dados de treino a partir de um arquivo binário
#Parametro rb lê o arquivo no modo binario
with open('/content/traffic-signs/data2.pickle', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # Carrega o arquivo pickle

# Preparando os rótulos de treino e validação para serem usados no Keras´
# Converte para formato categórico porque há 43 classes possiveis para cada imagem
data['y_train'] = to_categorical(data['y_train'], num_classes=43)   
data['y_validation'] = to_categorical(data['y_validation'], num_classes=43) #

# Alterando a ordem dos canais (para TensorFlow: canais no final)
data['x_train'] = data['x_train'].transpose(0, 2, 3, 1) #Altera ordem dos canais com transpose
data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

# Mostrando informações sobre os dados carregados
for i, j in data.items():
    if i == 'labels':               #laço que itera sobre o dataset para verificar e exibir suas dimensões
        print(i + ':', len(j))
    else:
        print(i + ':', j.shape)

# Definindo tamanhos de filtros e inicializando os modelos
filters = [3, 5, 9, 13, 15, 19, 23, 25, 31]  # Tamanhos de filtro
model = [0] * len(filters)  # Inicializando a lista de modelos

# Criando os modelos e adicionando camadas
for i in range(len(model)):
    model[i] = Sequential()
    model[i].add(Conv2D(32, kernel_size=filters[i], padding='same', activation='relu', input_shape=(32, 32, 3)))
    model[i].add(MaxPool2D(pool_size=2))  # Camada de Pooling
    model[i].add(Flatten())  # Achata as dimensões
    model[i].add(Dense(500, activation='relu'))  # Camada totalmente conectada
    model[i].add(Dense(43, activation='softmax'))  # Saída com 43 classes
    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compila o modelo

# Definindo o escalonador de taxa de aprendizado
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
epochs = 5  # Número de épocas

# Inicializando histórico dos modelos
h = [0] * len(model) #cria uma lista de tamanho igual ao número de modelos no objeto

# Treinando os modelos e armazena no h.
for i in range(len(h)):
    h[i] = model[i].fit(data['x_train'], data['y_train'],  #chama o metodo fit do modelo i   
                        batch_size=5, epochs=epochs,
                        validation_data=(data['x_validation'], data['y_validation']),
                        callbacks=[annealer], verbose=0)
