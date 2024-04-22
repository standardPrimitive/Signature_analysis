from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO(r'yolov8m.pt')
model.fuse()

docs_path = 'Docs'
predict_path = 'Predicts/Signs'


#########################################################################################
# Получение подписей из загжрунного документа
def extract_signatures_from_documents(docs_path, predict_path):
    os.makedirs(predict_path, exist_ok=True)
    for img_file in os.listdir(docs_path):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(docs_path, img_file)
            results = model.predict(img_path, conf=0.19, iou=0.3, verbose=False) 
            base_name = os.path.splitext(os.path.basename(img_file))[0]  # Базовое имя файла без расширения
            counter = 1  # Счетчик для добавления номера к сохраняемым файлам
    
            for idx, result in enumerate(results):
                boxes = result.boxes.xyxy
                classes = result.boxes.cls.cpu().numpy()  # Метки классов для каждой области
                for box, cls in zip(boxes, classes):
                    if cls == 1:  # Если класс объекта - подпись
                        # Получаем координаты прямоугольной рамки
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        
                        # Вырезаем область с подписью из исходного изображения
                        signature_img = result.orig_img[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Сохраняем вырезанную подпись в отдельное изображение
                        signature_save_path = os.path.join(predict_path, f'{base_name}_sign_{counter}.jpg')
                        cv2.imwrite(signature_save_path, signature_img)
                        counter += 1

    return "Подписи успешно получены из документов."
#########################################################################################
# Сравнение с подписями сотрудника
def text_output_signatures(employee_name, employee_signatures_path, predicts_path):
    # Загружаем все изображения подписей сотрудника 
    employee_signatures = []
    for filename in os.listdir(employee_signatures_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            signature_img = cv2.imread(os.path.join(employee_signatures_path, filename), cv2.IMREAD_GRAYSCALE)
            employee_signatures.append((filename, signature_img))

    # Загружаем все изображения подписей с документов
    document_signatures = []
    for filename in os.listdir(predicts_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            signature_img = cv2.imread(os.path.join(predicts_path, filename), cv2.IMREAD_GRAYSCALE)
            document_signatures.append((filename, signature_img))

    # Создаем объект sift детектора
    sift = cv2.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
 
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    # Пройдемся по каждому документу
    for doc_filename, doc_signature_img in document_signatures:
        # Находим keypoints и descriptors с помощью SIFT для queryImage
        kp1, des1 = sift.detectAndCompute(doc_signature_img, None)

        # Инициализируем переменные для хранения лучшего совпадения
        best_match = None
        best_match_score = 0

        # Пройдемся по каждой подписи сотрудника
        for signature_filename, signature_img in employee_signatures:
            # Находим keypoints и descriptors с помощью SIFT для trainImage
            kp2, des2 = sift.detectAndCompute(signature_img, None)

            # Выполняем сопоставление между дескрипторами изображений
            matches = flann.knnMatch(des1, des2, k=2)

            # Необходимо отрисовать только хорошие совпадения, поэтому создадим маску
            matchesMask = [[0, 0] for _ in range(len(matches))]

            # Проверим соответствие по критерию Лоу
            good_matches = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.57 * n.distance:
                    matchesMask[i] = [1, 0]
                    good_matches += 1

            # Обновляем лучшее совпадение, если текущее лучше
            if good_matches > best_match_score:
                best_match_score = good_matches
                best_match = signature_filename

        # Отрисовываем только хорошие совпадения
        draw_params = dict(matchColor=(0, 255, 0),
                            singlePointColor=(255, 0, 0),
                            matchesMask=matchesMask,
                            flags=cv2.DrawMatchesFlags_DEFAULT)    
        img3 = cv2.drawMatchesKnn(doc_signature_img, kp1, signature_img, kp2, matches, None, **draw_params)

        # Определяем пороговое значение для совпадения
        match_threshold = 10

        # Проверяем, превышает ли количество совпадающих особых точек пороговое значение
        res = ""

        if best_match_score >= match_threshold:
            res += (f"Подпись на документе {doc_filename} СОВПАДАЕТ с подписью сотрудника {employee_name} ({best_match}). \nКоличество совпадающих особых точек: {best_match_score}\n")
        else:
            res += (f"Подпись на документе {doc_filename} НЕ СОВПАДАЕТ с подписями сотрудника {employee_name}. \nКоличество совпадающих особых точек: {best_match_score}\n")
    
    return res
#########################################################################################
def visualize_best_signatures(employee_signatures_path, predicts_path):
    # Загружаем все подписи сотрудника 
    employee_signatures = []
    for filename in os.listdir(employee_signatures_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            signature_img = cv2.imread(os.path.join(employee_signatures_path, filename), cv2.IMREAD_GRAYSCALE)
            employee_signatures.append((filename, signature_img))

    # Загружаем все подписи из документов
    document_signatures = []
    for filename in os.listdir(predicts_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            signature_img = cv2.imread(os.path.join(predicts_path, filename), cv2.IMREAD_GRAYSCALE)
            document_signatures.append((filename, signature_img))

    # Создаем объект sift детектора
    sift = cv2.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Пройдемся по каждому документу
    for doc_filename, doc_signature_img in document_signatures:
        # Находим keypoints и descriptors с помощью SIFT для queryImage
        kp1, des1 = sift.detectAndCompute(doc_signature_img, None)

        # Переменные для отслеживания наилучшего соответствия
        best_signature_filename = None
        best_good_matches = 0
        best_matchesMask = []
        ###
        output_images = []
        ###
        # Пройдемся по каждой подписи сотрудника
        for signature_filename, signature_img in employee_signatures:
            # Находим keypoints и descriptors с помощью SIFT для trainImage
            kp2, des2 = sift.detectAndCompute(signature_img, None)

            # Выполняем сопоставление между дескрипторами изображений
            matches = flann.knnMatch(des1, des2, k=2)

            # Необходимо отрисовать только хорошие совпадения, поэтому создадим маску
            matchesMask = [[0, 0] for _ in range(len(matches))]

            # Проверим соответствие по критерию Лоу
            good_matches = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.57 * n.distance:
                    matchesMask[i] = [1, 0]
                    good_matches += 1

            # Обновляем переменные, если текущая подпись имеет больше хороших совпадений
            if good_matches > best_good_matches:
                best_signature_filename = signature_filename
                best_good_matches = good_matches
                best_matchesMask = matchesMask

        # Отображаем только лучшие совпадения
        if best_signature_filename is not None and best_good_matches >= 10:
            signature_img = cv2.imread(os.path.join(employee_signatures_path, best_signature_filename), cv2.IMREAD_GRAYSCALE)
            kp2, des2 = sift.detectAndCompute(signature_img, None)
            matches = flann.knnMatch(des1, des2, k=2)

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=best_matchesMask,
                               flags=cv2.DrawMatchesFlags_DEFAULT)

            img3 = cv2.drawMatchesKnn(doc_signature_img, kp1, signature_img, kp2, matches, None, **draw_params)
            ####
            output_path = os.path.join('output', f'{doc_filename}_matches.jpg')
            cv2.imwrite(output_path, img3)
            output_images.append(output_path)
            ###
            # print(f"Совпадения для документа {doc_filename} и подписи {best_signature_filename}: {best_good_matches} хороших совпадений")
            # plt.imshow(img3)
            # plt.title(f"Совпадения для документа {doc_filename} и подписи {best_signature_filename}")
            # plt.show()
        else:
            ###
            output_path = os.path.join('output', f'{doc_filename}_no_match.jpg')
            cv2.imwrite(output_path, doc_signature_img)
            output_images.append(output_path)
            ###
            # Если не найдено подходящей подписи, выведем подпись из документа и любую подпись сотрудника
            #print(f"Для документа {doc_filename} не найдено подходящей подписи. \nЛучшее совпадение: {best_signature_filename}, {best_good_matches} хороших совпадений")

            # Выведем подпись из документа
            # plt.imshow(doc_signature_img, cmap='gray')
            # plt.title(f"Совпадения для документа {doc_filename} и подписи {best_signature_filename}")
            # plt.show()
    return output_images
#########################################################################################