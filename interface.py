import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

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
        image = Image.open(image_path).resize((32, 32))  # Tamanho adequado ao modelo
        image_array = np.array(image) / 255.0  # Normalizar os valores
        image_array = image_array.reshape(1, 32, 32, 3)

        # Predição
        scores = model.predict(image_array)
        prediction = np.argmax(scores)
        return prediction, scores
    except Exception as e:
        messagebox.showerror("Erro", f"Falha na predição: {e}")
        return None, None

# Função para carregar imagem e exibir
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
            result_label.config(text=f"Predição: Classe {prediction}")

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
