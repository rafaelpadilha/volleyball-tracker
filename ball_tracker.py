import cv2
import pickle
from ultralytics import YOLO
import numpy as np
import logging
import torch
from ultralytics.engine.results import Masks
import random

class BallTracker():
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.log = logging.getLogger()

    def debug_frame_raw(self, frame):
        results = self.model(frame, conf=self.conf)
        if len(results) > 0:
            frame = results[0].plot()

        return frame
    
    def detect_ball(self, frame):
        balls = []

        results = self.model(frame, conf=self.conf, verbose=False, stream=True)

        for result in results:
            boxes = result.boxes  # caixas detectadas

            for i in range(len(boxes.cls)):
                cls = int(boxes.cls[i])  # classe detectada (ex: 0 = ball)
                conf = float(boxes.conf[i])  # confiança
                x1, y1, x2, y2 = boxes.xyxy[i]  # coordenadas da caixa

                if cls == 0:  # supondo que 0 = bola
                    balls.append({
                        "class": cls,
                        "conf": conf,
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                    })

        # Se houver bolas, retorna a de maior confiança
        if balls:
            return max(balls, key=lambda b: b["conf"])
        else:
            return []
