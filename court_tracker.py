import cv2
import pickle
from ultralytics import YOLO
import numpy as np
import logging
import torch
from ultralytics.engine.results import Masks
import random

class CourtTracker():
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.log = logging.getLogger()

    def detect_frame_raw(self, frame):
        results = self.model(frame)
        annotated_frame = results[0].plot()

        return annotated_frame

    def detect_frame_boxes(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False, stream=True)

        # Pegue imagem original
        original_img = results[0].orig_img.copy()

        lines = []

        for result in results:
            # Plot boxes
            for box in result.boxes:
                # Coords no formato xyxy (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Classe e confiança
                cls = self.model.names[int(box.cls[0])]
                conf = box.conf[0]

                if cls == 'Line':
                    lines.append((x1, y1, x2, y2))

                label = f"{cls} {conf:.2f}"

                # Desenha na imagem original
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if len(lines) == 2:
            self.log.debug('2 Linhas encontradas. Calculando distancia')
            centers = []
            for line in lines:
                x1, y1, x2, y2 = line
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))

            c1, c2 = centers
            distance = int(np.linalg.norm(np.array(c1) - np.array(c2)))
            cv2.line(original_img, c1, c2, (0, 255, 0), 1)

            # Escreve a distância na metade da linha
            mid_point = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            cv2.putText(
                original_img,
                f"{distance}px",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return original_img

    def detect_frame_mask(self, frame):
        results = self.model(frame)

        yolo_classes = list(self.model.names.values())
        classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
        colors = [(0, 0, 255),(0, 255, 0)]

        # Pegue imagem original
        original_img = results[0].orig_img.copy()

        for result in results:
            if result.masks:
                masks = result.masks.xy
                for box, mask in zip(result.boxes, masks):
                    points = np.int32([mask])
                    color_number = classes_ids.index(int(box.cls[0]))
                    cv2.fillPoly(original_img, points, colors[color_number])

                alpha = 0.25  # transparência da máscara
                cv2.addWeighted(original_img, alpha, frame, 1 - alpha, 0, frame)
        return frame
    
    def detect_frame_mask2(self, frame):
        results = self.model(frame)

        yolo_classes = list(self.model.names.values())
        class_ids = [yolo_classes.index(clas) for clas in yolo_classes]
        original_img = results[0].orig_img.copy()

        vertical_centers = []

        for result in results:
            if result.masks:
                masks = result.masks.xy

                for box, mask in zip(result.boxes, masks):
                    cls_id = int(box.cls[0])
                    if cls_id != 0:  # Considera apenas classe 0 (linhas)
                        continue

                    mask_np = np.array(mask, dtype=np.int32)
                    mask_np = mask_np.reshape(-1, 2)

                    # Calcula a linha central vertical ao longo da direção do segmento
                    rect = cv2.minAreaRect(mask_np)
                    box_pts = cv2.boxPoints(rect)
                    box_pts = np.array(box_pts, dtype=np.int32)

                    # Ordena por y para achar os dois pontos mais altos e baixos (para traçar linha vertical)
                    box_pts = sorted(box_pts, key=lambda p: p[1])
                    top_pts = box_pts[:2]
                    bottom_pts = box_pts[2:]

                    # Calcula centro da linha em X
                    top_center = np.mean(top_pts, axis=0)
                    bottom_center = np.mean(bottom_pts, axis=0)

                    # Reduz 5% da linha nas pontas (suavização)
                    line_vec = bottom_center - top_center
                    reduction = 0.00
                    shrink_vec = line_vec * reduction
                    top_center = top_center + shrink_vec
                    bottom_center = bottom_center - shrink_vec

                    top_center = tuple(map(int, top_center))
                    bottom_center = tuple(map(int, bottom_center))

                    # Salva para cálculo da distância depois
                    vertical_centers.append((top_center, bottom_center))

                    # Desenha a linha vertical (verde)
                    cv2.line(original_img, top_center, bottom_center, (0, 255, 0), thickness=3)

        if len(vertical_centers) == 2:
            # Desenha linhas horizontais entre as extremidades
            pt1_top, pt2_top = vertical_centers[0][0], vertical_centers[1][0]
            pt1_bot, pt2_bot = vertical_centers[0][1], vertical_centers[1][1]

            # Linha superior (azul)
            cv2.line(original_img, pt1_top, pt2_top, (255, 0, 0), 2)
            dist_top = int(np.linalg.norm(np.array(pt1_top) - np.array(pt2_top)))
            mid_top = tuple(((np.array(pt1_top) + np.array(pt2_top)) // 2).astype(int))
            cv2.putText(original_img, f"{dist_top}px", mid_top, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Linha inferior (azul)
            cv2.line(original_img, pt1_bot, pt2_bot, (255, 0, 0), 2)
            dist_bot = int(np.linalg.norm(np.array(pt1_bot) - np.array(pt2_bot)))
            mid_bot = tuple(((np.array(pt1_bot) + np.array(pt2_bot)) // 2).astype(int))
            cv2.putText(original_img, f"{dist_bot}px", mid_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return original_img



