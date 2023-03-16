# Importar bibliotecas necesarias
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import preparar_set
import set_de_test
# import imagen_a_texto

tamanyo_max = 128

# Definir los datos de entrada
data = preparar_set.preparar_set()

# Cargar el modelo y el tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocesar los datos de entrada
sentences = [x[0] for x in data]
labels = [x[1] for x in data]

encoded_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=tamanyo_max, return_tensors='pt')

# Convertir las etiquetas a tensores de PyTorch
label_dict = {"factura": 0, "nota de gasto": 1, "albarán": 2}
labels = [label_dict[label] for label in labels]
labels = torch.tensor(labels)

# Entrenar el modelo
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(3):
    outputs = model(**encoded_inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Realizar predicciones con el modelo entrenado
model.eval()


# Guardar el modelo entrenado
output_dir = './modelo-entrenado'
model.save_pretrained(output_dir)

print ("MODELO ENTRENADO ")



# test_sentence = imagen_a_texto.getTextoDeImagen("./set_docs/Pruebas/factura_de_prueba.PNG")

# inputs = tokenizer(test_sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')
# outputs = model(**inputs)
# predictions = torch.argmax(outputs.logits, dim=1)

# label_dict = {0: "factura", 1: "nota de gasto", 2: "albarán"}

# Convertir los tensores de predicción en etiquetas
# predicted_labels = [label_dict[prediction.item()] for prediction in predictions]


# print("PREDICCIÓN")
# print(predicted_labels)
