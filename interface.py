import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

# Função para carregar o modelo
def load_prediction_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao carregar o modelo: {e}")
        return None

# Função para processar e fazer predição
def predict_image(model, image_path):
    try:
        # Carregar e preprocessar imagem
        target_size=(32, 32)
        image = Image.open(image_path)  # Abrir a imagem
        image = image.resize(target_size)  # Redimensionar para 32x32
        image = img_to_array(image)  # Converter para array NumPy
        image = image[:, :, :3]  # Garantir que há apenas 3 canais (RGB)
        image = image / 255.0  # Normalizar os valores dos pixels (em hexa)
        image = np.expand_dims(image, axis=0)  # Adicionar dimensão de batch

        # Predição
        scores = model.predict(image)
        prediction = np.argmax(scores) #argmax pega a classe com maior valor e armazena no prediction
        return prediction, scores
    except Exception as e:
        messagebox.showerror("Erro", f"Falha na predição: {e}")
        return None, None
#Lê o csv
def label_text(file):
    r = pd.read_csv(file)
    return list(r['SignName'])

# Função para carregar imagem e exibir predição
def load_and_display_image():
    file_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Exibir imagem
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Reduzir tamanho para exibição
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predizer
        prediction, scores = predict_image(model, file_path)
        if prediction is not None:
            #result_label.config(text=f"Predição: Classe {prediction}")
            # Carregar os rótulos
            labels = label_text('label_names.csv')

            # Mostrar o rótulo previsto
            print('Label:', labels[prediction])
            result_label.config(text=f"Predição: Classe {labels[prediction]}")

# Janela principal
root = tk.Tk()
root.title("Classificação de Imagens")

# Carregar modelo de predição
model = load_prediction_model('model-3x3.h5')

# Interface
frame = tk.Frame(root)
frame.pack(pady=20)

load_button = tk.Button(frame, text="Carregar Imagem", command=load_and_display_image)
load_button.pack(pady=10)

image_label = tk.Label(frame)
image_label.pack(pady=10)

result_label = tk.Label(frame, text="Resultado: Nenhum")
result_label.pack(pady=10)

root.mainloop()
