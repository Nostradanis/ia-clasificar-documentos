import os
import imagen_a_texto


def getTextos(ruta, etiqueta):
    archivos = os.listdir(ruta)
    objetos = [] 
    # Imprimir los nombres de los archivos encontrados
    for archivo in archivos:
        o = (imagen_a_texto.getTextoDeImagen(ruta + archivo), etiqueta)
        print("---------------------------------------------------")
        print(o)
        print("---------------------------------------------------")
        objetos.append( o )
    
    return objetos[0]

def preparar_set():
    ruta_facturas = "./set_docs/Facturas/"
    ruta_albaranes = "./set_docs/Facturas/"
    ruta_notas = "./set_docs/Facturas/"

    set_docs = []
    set_docs.append(getTextos(ruta_facturas,"factura"))
    set_docs.append(getTextos(ruta_albaranes,"albarán"))
    set_docs.append(getTextos(ruta_notas,"nota de gasto"))

    print ("SET PREPARADO")
    #print (set_docs)

    return set_docs

