# Importar bibliotecas necesarias
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import imagen_a_texto

tamanyo_max = 128
prueba = "nota"

# Cargar el modelo guardado
model_path = './modelo-entrenado'
model = BertForSequenceClassification.from_pretrained(model_path)

# Cargar el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Definir los datos de entrada

data = [(imagen_a_texto.getTextoDeImagen("./set_docs/Pruebas/"+prueba+"_de_prueba.PNG"),"")]
#data = [("factura de la sardanya","")]

# Preprocesar los datos de entrada
input_ids = []
attention_masks = []
for sent, label in data:
    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True,truncation=True, max_length=tamanyo_max, return_attention_mask=True, return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])   
    attention_masks.append(encoded_dict['attention_mask'])
   
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Realizar la inferencia
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    _, predictions = torch.max(outputs[0], dim=1)

# Definir el diccionario de etiquetas
label_dict = {0: "factura", 1: "nota de gasto", 2: "albarán"}

# Convertir los tensores de predicción en etiquetas
predicted_labels = [label_dict[prediction.item()] for prediction in predictions]

# Imprimir la etiqueta predicha
print(predicted_labels[0])