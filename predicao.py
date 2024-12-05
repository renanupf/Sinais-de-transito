import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Carregar e preparar a imagem
def prepare_image(image_path, target_size=(32, 32)):
    image = Image.open(image_path)  # Abrir a imagem
    image = image.resize(target_size)  # Redimensionar para 32x32
    image = img_to_array(image)  # Converter para array NumPy
    image = image[:, :, :3]  # Garantir que há apenas 3 canais (RGB)
    image = image / 255.0  # Normalizar os valores dos pixels
    image = np.expand_dims(image, axis=0)  # Adicionar dimensão de batch
    return image

# Caminho para a imagem de teste
image_path = 'teste.png'
x_input = prepare_image(image_path)

# Mostrar a imagem carregada
plt.imshow(plt.imread(image_path))
plt.axis('off')
plt.show()

# Carregar o modelo
model = load_model('model-3x3.h5')

# Fazer a predição
scores = model.predict(x_input)  # Realiza a predição
print(f"Formato das previsões: {scores.shape}")

# Obter a classe prevista
prediction = np.argmax(scores)
print('ClassId:', prediction)

# Função para obter os rótulos das classes
def label_text(file):
    r = pd.read_csv(file)
    return list(r['SignName'])

# Carregar os rótulos
labels = label_text('/content/traffic-signs/label_names.csv')

# Mostrar o rótulo previsto
print('Label:', labels[prediction])
