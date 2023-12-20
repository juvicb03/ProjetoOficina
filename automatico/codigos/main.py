import os
import cv2
import numpy as np
import pytesseract as pyt
import psycopg2 as post
import util

# modelo e peso para o yolo
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

input_dir = 'C:/Users/USER/PycharmProjects/pythonProject/automatico/data'

# conexao com o banco de dados
conn = post.connect(database="oficina", host="localhost", user="postgres", password="joca", port="5432")

for img_name in os.listdir(input_dir):

    img_path = os.path.join(input_dir, img_name)

    # carregar o modelo, a imagem e relizar a detecçao
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    img = cv2.imread(img_path)
    H, W, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    net.setInput(blob)
    detections = util.get_outputs(net)

    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    plate_num = ""
    # relizar o pre processamento, o ocr e formatar a string de saida
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)  # transformar para cinza
        # cv2.imshow("Gray", gray)
        # cv2.waitKey(0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # binarizarçao otsu
        # cv2.imshow("Otsu", thresh)
        # cv2.waitKey(0)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # transformaçao morfologica
        dilation = cv2.dilate(thresh, rect_kern, iterations=1)  # dilataçao
        # cv2.imshow("Dilatacao", dilation)
        # cv2.waitKey(0)
        try:
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        im2 = gray.copy()

        # loop nos contornos realizando ocr
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width = im2.shape

            if height / float(h) > 6: continue
            ratio = h / float(w)
            if ratio < 1.5: continue
            area = h * w
            if width / float(w) > 25: continue
            if area < 100: continue
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)  # remover ruido da imagem
            # cv2.imshow("Digito", roi)
            # cv2.waitKey(0)
            # relizaçao do ocr, obtendo apenas letras maiusculas e numeros
            text = pyt.image_to_string(roi,
                                       config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            text = text.strip()
            plate_num += text
        print(plate_num)

    # mapeamento entre letras e numeros, para a troca de digitos similantes
    mapeamento_letra_numero = {'O': '0', 'I': '1', 'E': '3', 'A': '4', 'S': '5', 'B': '8'}
    mapeamento_numero_letra = {'0': 'O', '1': 'I', '3': 'E', '4': 'A', '5': 'S', '8': 'B'}
    translation_letra = str.maketrans(mapeamento_letra_numero)
    translation_numero = str.maketrans(mapeamento_numero_letra)
    lista = list(plate_num)

    # tratamento para o formato da string, garantir que seja letra e numero nos lugares certos
    j = len(lista)
    i = 0
    while (i < j):
        if (i < 3 or i == 4):
            lista[i] = lista[i].translate(translation_numero)
        else:
            lista[i] = lista[i].translate(translation_letra)
        i += 1

    print(lista)
    final = ''.join(str(e) for e in lista)
    print(final)

    cursor = conn.cursor()

    # chamada da funçao criada no banco de dados
    tupla = cursor.callproc('verifica_carro', (final,))
    print(tupla)

    # Executing the SELECT query
    # cursor.execute("SELECT placa FROM carro WHERE placa=%s;", (final,))

    # recuperar a tupla e printar o resultado
    result = cursor.fetchall()
    for row in result:
        print(row[0])

    cursor.close()
conn.close()
