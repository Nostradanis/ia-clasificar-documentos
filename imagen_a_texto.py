import pytesseract
from PIL import Image
import re
from collections import Counter
from PIL import Image
import numpy as np
import torch

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

#La primera vez
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

def getImageArrayDeImagen(imagen):
    imagen = Image.open(imagen)
    imagen = imagen.convert('RGB')
    imagen = imagen.resize((224, 224))  # Reemplazar con el tamaño de entrada del modelo
    imagen_array = np.array(imagen)
    imagen_array = imagen_array / 255.0
    imagen_tensor = torch.tensor(imagen_array).unsqueeze(0)

    print(imagen_tensor)
    print("Tamaño: " + str(len(imagen_tensor)))
    


def getTextoDeImagen(imagen):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    img = Image.open(imagen)

    texto = pytesseract.image_to_string(img, lang='spa') 
    texto = re.sub(r'\s+', ' ', texto)
    #print(texto)
    #print("Tamaño: "+ str(len(texto)))
    return  limpiar_texto( eliminar_importes(texto) )

def eliminar_importes(texto):
    # Expresión regular para encontrar importes
    regex = r"\d+(?:[.,]\d+)*(?:\s*(?:€|\$|USD|EUR|MXN|GBP|JPY|CHF|CNY|HKD|CAD))?"
    
    # Buscar y eliminar los importes del texto
    texto_limpio = re.sub(regex, "", texto)
    
    return texto_limpio

def ordenar_palabras_por_ocurrencias(texto):
    # Separar el texto en palabras
    palabras = texto.split()

    # Contar la frecuencia de cada palabra
    frecuencia_palabras = Counter(palabras)

    # Ordenar las palabras por frecuencia
    palabras_ordenadas = sorted(frecuencia_palabras, key=frecuencia_palabras.get, reverse=True)

    # Crear un string con las palabras ordenadas
    resultado = " ".join(palabras_ordenadas)

    return resultado

def limpiar_texto(texto):
    # Tokenizar el texto
    tokens = word_tokenize(texto.lower(), language='spanish')
    
    # Eliminar las preposiciones y las palabras que no tienen sentido
    stop_words = set(nltk.corpus.stopwords.words('spanish'))
    limpio = [token for token in tokens if token.lower() not in stop_words and wn.synsets(token.lower(), lang='spa')]
    
    # Unir los tokens limpios en un string
    limpio = ' '.join(limpio)

    # Eliminar letras sueltas
    limpio = re.sub(r'\b\w\b', '', limpio)
    # Eliminar símbolos
    limpio = re.sub(r'[\\/*?¿"!¡\'{}-]', ' ', limpio)
    # Reemplazar múltiples espacios en blanco por uno solo
    limpio = re.sub(r'\s+', ' ', limpio)
    # Eliminar espacios en blanco al principio y final del texto
    limpio = limpio.strip()
    
    return limpio

import re




#getTextoDeImagen("./set_docs/Facturas/32346.PNG")

