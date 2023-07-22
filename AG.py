import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
import tensorflow as tf
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


#variables para graficar
media = []
max = []
min = []
individuosCsv = []

# Función para calcular el valor de la función objetivo
def redNeuronal(capas, neuronas, epocas, lnRate, momento):
    test = ["DatosFinal/Datos_Train(K1_Test_Random).csv","DatosFinal/Datos_Train(K2_Test_Random).csv","DatosFinal/Datos_Train(K3_Test_Random).csv"]

    train = ["DatosFinal/Datos_Train(K1_Train_Random).csv","DatosFinal/Datos_Train(K2_Train_Random).csv","DatosFinal/Datos_Train(K3_Train_Random).csv"]

    resul = []

    for i in range(0,3):
        print("K <<<<<<<<<<<<<<<<<<<" + str(i))
        lnRate = lnRate / 1000
        momento = momento / 100

        # Cargar los datos de entrenamiento desde el archivo CSV
        df_train = pd.read_csv(test[i])

        # Obtener las características de entrenamiento (X_train) y las etiquetas de entrenamiento (y_train)
        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values

        # Cargar los datos de prueba desde el archivo CSV
        df_test = pd.read_csv(train[i])

        # Obtener las características de prueba (X_test) y las etiquetas de prueba (y_test)
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

        # Codificar las etiquetas como variables categóricas
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = label_encoder.transform(y_test)
        y_test = to_categorical(y_test, num_classes=10)

        # Crear el modelo de la red neuronal
        model = Sequential()
        model.add(Dense(neuronas, activation='relu', input_shape=(X_train.shape[1],)))

        for n in range(capas - 2):
            model.add(Dense(neuronas, activation='relu'))

        model.add(Dense(10, activation='softmax'))

        # Configurar el optimizador
        opt = SGD(learning_rate=lnRate, momentum=momento)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Definir el callback para guardar los checkpoints del mejor modelo
        checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='accuracy', save_best_only=True, mode='max')
        # Definir el callback para el early stopping basado en la precisión
        early_stopping = EarlyStopping(monitor='accuracy', patience=10)

        # Entrenar el modelo
        history = model.fit(X_train, y_train, batch_size=1200, epochs=epocas, callbacks=[checkpoint, early_stopping], validation_data=(X_test, y_test))

        # Extraer los valores de precisión por época
        accuracy_values = history.history['accuracy']

        resul.append(accuracy_values[-1])

    mediaResul = np.mean(resul)
    return mediaResul




# Función para convertir un número en su representación binaria con un número fijo de bits
def to_binary(num, num_bits):
    binary = bin(num)[2:]
    padding = '0' * (num_bits - len(binary))
    print(padding + binary)
    return padding + binary

# Función para convertir un número binario en su equivalente decimal
def to_decimal(binary):
    return int(binary, 2)

poblacion = []
numero_individuos = 10
longitud_bits = 32

while len(poblacion) < numero_individuos:
    individuo = ""
    for _ in range(longitud_bits):
        bit = random.choice(["0", "1"])
        individuo += bit

    decimal_0_3 = int(individuo[:4], 2)
    decimal_4_8 = int(individuo[4:9], 2)
    decimal_9_15 = int(individuo[9:16], 2)
    decimal_16_24 = int(individuo[16:25], 2)
    decimal_25_31 = int(individuo[25:], 2)

    if 1 <= decimal_0_3 <= 10 and 4 <= decimal_4_8 <= 25 and 1 <= decimal_9_15 <= 200 \
                and 1 <= decimal_16_24 <= 300 and 1 <= decimal_25_31 <= 100:
        poblacion.append({"individuo": individuo, "calificacion": 0})
xG = []

# Ciclo de evolución (cambiando el range cambia el numero de ciclos)
for generation in range(60):
    xG.append(generation)
    print("generacion: " + str(generation))
    # Imprimir la población inicial
    # for individuo in poblacion:
    #    print(f"Individuo: {individuo['individuo']}, Calificación: {individuo['calificacion']}")

    # Creación de nuevos individuos mediante reproducción
    #print("hijos")
    for i in range(0, len(poblacion), 2):
        corte1 = random.randint(1, len(poblacion)-1)
        corte2 = random.randint(1, len(poblacion)-1)
        #print(i)
        # Selecciona los padres del menor al mayor
        parent1 = poblacion[i]
        parent2 = poblacion[i+1]
                    #[1100][01010]            +      [1100110]                      +       [001100110][1100110]
        temp1 = parent1['individuo'][:corte1] + parent2['individuo'][corte1:corte2] + parent1['individuo'][corte2:]
        temp2 = parent2['individuo'][:corte1] + parent1['individuo'][corte1:corte2] + parent2['individuo'][corte2:]
        #print("padre: " + parent1['individuo'])
        #print("padre: " + parent2['individuo'])


        #   
        # [0011][10101]   [0010110]  [001100110][1100110]
        
        child1 = temp1[0:2] + temp2[2:5] + temp2[5:6] + temp1[6:7] + temp2[7:8] + temp1[8:9] + temp2[9:10] + temp1[10:12] + temp2[12:14] + temp1[14:16] + temp2[16:17] + temp2[17:19] + temp1[19:21] + temp2[21:23] + temp1[23:25] + temp2[25:26] + temp1[26:28] + temp2[28:30] + temp1[30:32] + temp2[32:33]

        # invertido
        child2 = temp2[0:2] + temp1[2:5] + temp1[5:6] + temp2[6:7] + temp1[7:8] + temp2[8:9] + temp1[9:10] + temp2[10:12] + temp1[12:14] + temp2[14:16] + temp1[16:17] + temp1[17:19] + temp2[19:21] + temp1[21:23] + temp2[23:25] + temp1[25:26] + temp2[26:28] + temp1[28:30] + temp2[30:32] + temp1[32:33]
        
        #print("hijo : " + child1)
        #print("hijo : " + child2)

        poblacion.append({"individuo": child1, "calificacion": 0})
        poblacion.append({"individuo": child2, "calificacion": 0})


    # Seleccionar el individuo deseado para la mutación
    individuo_index = random.randint(0, len(poblacion)-1)  # Índice del individuo que deseas mutar


    # Realizar la mutación en bits específicos
    individuo_mutado = list(poblacion[individuo_index]['individuo'])
    mutCapa = random.randint(0,3)
    mutNeu = random.randint(4,8)
    mutEpo = random.randint(9,15)
    mutLR = random.randint(16,24)
    mutMo = random.randint(24,31)
    bits_a_mutar = [mutCapa, mutNeu, mutEpo, mutLR, mutMo]  # Índices de los bits que deseas mutar
    #print("bits a mutar: " + str(mutCapa) + " " + str(mutNeu) + " " + str(mutEpo) + " " + str(mutLR) + " " + str(mutMo))
    # Imprimir el individuo antes de la mutación
    #print(f"Individuo antes de la mutación  : {poblacion[individuo_index]['individuo']}")

    for bit_index in bits_a_mutar:
        if individuo_mutado[bit_index] == "0":
            individuo_mutado[bit_index] = "1"
        else:
            individuo_mutado[bit_index] = "0"

    # Actualizar el individuo mutado en la población
    poblacion[individuo_index]['individuo'] = "".join(individuo_mutado)

    # Imprimir el individuo después de la mutación
    #print(f"Individuo después de la mutación: {poblacion[individuo_index]['individuo']}")

    # filtrar poblacion
    poblacion_valida = []

    for individuo in poblacion:
        if individuo['calificacion'] == 0:
            # Obtener los valores decimales correspondientes a los rangos de bits
            decimal_0_3 = int(individuo['individuo'][:4], 2)
            decimal_4_8 = int(individuo['individuo'][4:9], 2)
            decimal_9_15 = int(individuo['individuo'][9:16], 2)
            decimal_16_24 = int(individuo['individuo'][16:25], 2)
            decimal_25_31 = int(individuo['individuo'][25:], 2)

            # Verificar las condiciones para cada rango de bits
            if 1 <= decimal_0_3 <= 10 and 4 <= decimal_4_8 <= 25 and 1 <= decimal_9_15 <= 200 \
                and 1 <= decimal_16_24 <= 300 and 1 <= decimal_25_31 <= 100:
                #print ("individuo aprobado")
                individuo['calificacion'] = redNeuronal(decimal_0_3, decimal_4_8, decimal_9_15, decimal_16_24, decimal_25_31)
                poblacion_valida.append(individuo)
        else:
            poblacion_valida.append(individuo)

    poblacion_top_10 = []

    poblacion_ordenada = sorted(poblacion_valida, key=lambda x: x['calificacion'], reverse=True)
    poblacion_top_10 = poblacion_ordenada[:10]

    # Obtener las calificaciones de los individuos en poblacion_top_10
    calificaciones = np.array([individuo['calificacion'] for individuo in poblacion_top_10])

    # Calcular la media, el valor máximo y el valor mínimo
    media.append(np.mean(calificaciones))
    max.append(np.max(calificaciones))
    min.append(np.min(calificaciones))
    # Imprimir la población ordenada y recortada
    temp = 1
    for individuo in poblacion_top_10:
        individuosCsv.append({"id": temp,"individuo": individuo['individuo'], "calificacion": individuo['calificacion'], "generacion": generation})
        temp = temp + 1
    #borramos arreglos para liberar memoria
    poblacion = []
    poblacion = poblacion_top_10


ruta_archivo_csv = 'individuos2.csv'

# Lista de encabezados de las columnas
encabezados = ['id', 'individuo', 'calificacion', 'generacion']

# Guardar los datos en un archivo CSV
with open(ruta_archivo_csv, 'w', newline='') as archivo_csv:
    escritor_csv = csv.DictWriter(archivo_csv, fieldnames=encabezados)

    # Escribir los encabezados en la primera línea
    escritor_csv.writeheader()

    # Escribir los datos de cada individuo
    for individuo in individuosCsv:
        escritor_csv.writerow(individuo)

print('Datos guardados en', ruta_archivo_csv)
# Imprimir los resultados
plt.plot(xG, min, label='minimo')
plt.plot(xG, max, label='maximo')
plt.plot(xG, media, label='media')
plt.xlabel('numero de ciclos')
plt.ylabel('accuracy')
plt.title('Gráfica')
plt.legend()
plt.show()



