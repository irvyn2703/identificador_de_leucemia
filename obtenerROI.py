import cv2
import numpy as np
import os


def resize_image(image, width, height):
    # Obtener las dimensiones originales de la imagen
    original_height, original_width = image.shape[:2]

    # Calcular el factor de escala para redimensionar la imagen
    scale_factor = min(width / original_width, height / original_height)

    # Calcular las nuevas dimensiones
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Redimensionar la imagen utilizando la interpolación bilineal
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image


def segment_image(image_path):
    # accede al numero de Roi's
    global roiCounter

    # Cargar la imagen
    image = cv2.imread(image_path)

    # Redimensionar la imagen a un tamaño específico
    resized_image = resize_image(image, 800, 600)

    # Convertir la imagen redimensionada a escala de grises
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detección de color para segmentar células infectadas
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    # Rango inferior y superior de color morado
    # lower_color = np.array([125, 50, 50])
    # upper_color = np.array([150, 255, 255])
    lower_color = np.array([125, 50, 50])
    upper_color = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Encontrar los contornos de las regiones de interés
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se establece un rango de area para filtar los contornos encontrados
    min_area = 500    # Área mínima de un objeto
    max_area = 10000  # Área máxima de un objeto

    # Filtra los contornos en base al rango de area Especificado
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            filtered_contours.append(contour)
    #   output = cv2.drawContours(seg_img, filtered_contours, -1, (0, 0, 255), 2)

    # Crear una lista para almacenar los rois
    rois = []

    # SI los contornos de la imagen son mas de 10 solo selecciona los primeros 10
    # if len(filtered_contours) >= 10:
    #     filtered_contours = filtered_contours[:11]

    # Valores de ajuste del recorte de los ROIs
    offset = 10
    expansion = 20
    # Iterar sobre los contornos y obtener los rois
    for contour in filtered_contours:
        # Obtener el rectángulo del contorno con un pequeño desplazamiento
        x, y, w, h = cv2.boundingRect(contour)
        x -= 10
        y -= 20
        w += 25
        h += 35
        # x -= 10
        # y -= 10
        # w += 15
        # h += 25

        # Asegurarse de que los valores no sean negativos o excedan las dimensiones de la imagen
        x = max(0, x)
        y = max(0, y)
        w = min(w, resized_image.shape[1] - x)
        h = min(h, resized_image.shape[0] - y)

        # Añadir el roi a la lista
        roi = resized_image[y:y + h, x:x + w]
        rois.append(roi)

        # Dibujar un rectángulo alrededor del roi en la imagen original
        # cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Guardar el roi como una imagen individual
        # cv2.imwrite(f'roi_{roiCounter}.jpg', roi)
        # cv2.imwrite(f"imagenes/roi_{roiCounter}.jpg", roi)
        # cv2.imwrite(f"Train_ROI/ALL_train/roi_{roiCounter}.jpg", roi)
        # cv2.imwrite(f"Train_ROI/AML_train/roi_{roiCounter}.jpg", roi)
        # cv2.imwrite(f"Train_ROI/CLL_train/roi_{roiCounter}.jpg", roi)
        # cv2.imwrite(f"Train_ROI/CML_train/roi_{roiCounter}.jpg", roi)
        # cv2.imwrite(f"ROI2_roi/CML/roi_{roiCounter}.jpg", roi)
        cv2.imwrite(f"ROI2_roi/CLL/roi_{roiCounter}.jpg", roi)
        roiCounter += 1

    # Mostrar la imagen redimensionada con los rois
    # cv2.imshow('Resized Image with ROIs', resized_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rois




# Segmentar cada imagen en la carpeta
for image_file in image_files:
    # Ruta completa de la imagen
    image_path = os.path.join(folder_path, image_file)

    # Segmentar la imagen y obtener los rois
    rois = segment_image(image_path)