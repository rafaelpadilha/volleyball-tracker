import logging
import cv2
import numpy as np
import math
import time
import torch
from collections import deque

from court_tracker import CourtTracker
from ball_tracker import BallTracker


CLEAN_DEQUE_TIME = 5
FRAME_RES = (1280, 720)

class VideoProcessor:
    def __init__(self, logger: logging.RootLogger):
        self.log = logger
        self.log.debug('Iniciando modulo de processamento')

        # Calibrar court_detector, usando marcas manuais por enquanto
        # self.court_detector = CourtTracker('./models/court_detector_v2.pt', conf=0.5)

        # self.ball_detector = BallTracker('./models/yolo11m-volley.pt', conf=0.65)
        self.ball_detector = BallTracker('./models/YOLOv12s.pt', conf=0.6) # Melhor até agora
        # self.ball_detector = BallTracker('./models/ball_detector_v2.pt', conf=0.6)
        self.stream = False
        self.fps = None

        self.court_points = []
        self.v_max = None

        self.trajectory = deque(maxlen=5)
        self.last_update = time.time()

        if torch.cuda.is_available():
            self.log.info('Using GPU!')
        else:
            self.log.warning('Using CPU!!!')

        try:
            # self._load_video('data/video.mkv', skip=11940.0)
            # self._load_video('data/pt.mp4')
            self._load_video('data/IMG_1001.MP4')
            # self._load_cam()
            self._show_video()
        except Exception as e:
            raise e
        finally:
            self._exit()

    def _load_video(self, video_path, skip=None) -> None:
        # Abrir o vídeo
        self.cap = cv2.VideoCapture(video_path)

        # Pular Quadros
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, skip)

        # Verificar se abriu corretamente
        if self.cap.isOpened():
            self.log.info("Video carregado com sucesso.")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.log.info(f"FPS: {self.fps}")
        else:
            self.log.error("Erro ao abrir o vídeo.")
            raise Exception('Falha ao ler vídeo')
        
    def _load_cam(self) -> None:
        # Abrir o vídeo
        self.cap = cv2.VideoCapture(0)

        # Verificar se abriu corretamente
        if self.cap.isOpened():
            self.log.info("Video carregado com sucesso.")
        else:
            self.log.error("Erro ao abrir o vídeo.")
            raise Exception('Falha ao ler vídeo')
        
    def _show_video(self):
        self.log.info('Iniciando exibição do vídeo')

        # Calcula o delay para mostrar caso seja vídeo
        px_per_m = None
        if self.stream:
            delay_ms = 1
        else:
            delay_ms = int(1000 / self.fps)

        self.log.debug(f"Delay: {delay_ms} ms")

        # Loop de processamento
        while True:
            ret, frame = self.cap.read()

            if not ret:
                self.log.warning('Não obteve mais frames, saindo.')
                break  # Fim do vídeo
            
            # Facilita a visualização
            frame = cv2.resize(frame, FRAME_RES)

            if not px_per_m:
                px_per_m = self.calibrar_pixels_por_metro(frame)
                # px_per_m = 50 # DEBUG

            # Detecta a bola com maior confiança no frame
            ball = self.ball_detector.detect_ball(frame)

            if ball:
                # Adicionar bola na trajetoria
                x1, y1, x2, y2 = map(int, ball["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1) # Plota a caixa da bola detectada
                ball["time"] = time.time()

                self.trajectory.append(ball)
                self.last_update = time.time()

                # Se for a segunda bola da trajetória calucular distância e velocidade
                if len(self.trajectory) >= 2:
                    b1 = self.trajectory[-2]
                    b2 = self.trajectory[-1]

                    b1x, b1y = map(int, b1["center"])
                    cv2.circle(frame, (b1x, b1y), 3, (0,0,255), -1) # Plota um circulo na bola anterior
                    b2x, b2y = map(int, b2["center"])

                    delta_s = b2["time"] - b1["time"] # Tempo entre as detecções 

                    dist_px = math.hypot(b2x - b1x, b2y - b1y)  # Distância em px
                    dist_m = dist_px / px_per_m                 # Convertendo para metros
                    vel_kmh = (dist_m / delta_s) * 3.6          # Converte de ms/h -> km/h

                    # Salva temporáriamente a maior velocidade registrada
                    if not self.v_max:
                        self.v_max = vel_kmh
                    else:
                        self.v_max = max(self.v_max, vel_kmh)

                    cv2.putText(frame, f"Vel: {vel_kmh:.2f} km/h", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.v_max:
                cv2.putText(frame, f"Vmax: {self.v_max:.2f} km/h", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Limpa a trajetória eventualmente para que não acumule dados desnecessários
            if time.time() - self.last_update > CLEAN_DEQUE_TIME:
                    self.log.debug('Limpando trajetória, pois não houve atualizações recentes.')
                    self.trajectory.clear()
                    self.v_max = None

            # Versão antiga que marcava a tragetória 
            # if len(self.trajectory) > 1:
            #     distancia = None
            #     velocidade = None
            #     velocidade_max = None
            #     for b1, b2 in zip(self.trajectory, list(self.trajectory)[1:]):
            #         b1x, b1y = map(int, b1["center"])
            #         cv2.circle(frame, (b1x, b1y), 5, (0,0,255), -1)
            #         b2x, b2y = map(int, b2["center"])
            #         cv2.circle(frame, (b2x, b2y), 5, (0,0,255), -1)

            #         cv2.line(frame, (b1x, b1y), (b2x, b2y), color=(0, 255, 0), thickness=2)

            #         distancia = math.hypot(b2x - b1x, b2y - b1y)
            #         velocidade = distancia / (b2["time"] - b1["time"]) if (b2["time"] - b1["time"]) > 0 else 0
            #         if not velocidade_max:
            #             velocidade_max = velocidade
            #         else:
            #             velocidade_max = max(velocidade_max, velocidade)

            #     cv2.putText(frame, f"Vel: {velocidade:.2f} px/s", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            #     cv2.putText(frame, f"Vmax: {velocidade_max:.2f} px/s", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            #     if time.time() - self.last_update > CLEAN_DEQUE_TIME:
            #         self.trajectory.clear()

            # Debugar modelos de detecção: 

            # frame = self.court_detector.detect_frame_boxes(frame)
            # frame = self.court_detector.detect_frame_raw(frame)
            # frame = self.ball_detector.debug_frame_raw(frame)

            # Exibir o frame
            cv2.imshow("Video", frame)

            # Aguarda o delay para ir para o próximo quadro
            # Se o usuário apertar "q" sai do programa
            if cv2.waitKey(delay_ms) == ord('q'):
                break

        # Liberar recursos

    def mouse_click(self, event, x, y, flags, param):  
        if event == cv2.EVENT_LBUTTONDOWN:
            self.court_points.append((x, y))
            self.log.debug(f"Ponto {len(self.court_points)}: ({x}, {y})")

            # Desenha o clique no frame
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)

            # Se for o segundo ou o quarto clique, desenha uma linha
            if len(self.court_points) in (2, 4):
                cv2.line(param, self.court_points[-2], self.court_points[-1], (0, 255, 0), 2)

            cv2.imshow("calibragem", param)

    def calibrar_pixels_por_metro(self, frame) -> float:
        """
        Recebe um frame (BGR) e permite que o usuário clique 4 pontos:
        - P1, P2 -> extremidades da primeira linha de 3 metros
        - P3, P4 -> extremidades da segunda linha de 3 metros

        Retorna o valor de pixels por metro (média das duas medições).
        """
        self.court_points.clear()

        clone = frame.copy()
        cv2.namedWindow("calibragem")
        cv2.imshow("calibragem", clone)
        cv2.setMouseCallback("calibragem", self.mouse_click, clone)

        self.log.info("Clique nas extremidades das duas linhas de 3 metros (4 cliques no total).")
        self.log.info("Ordem: P1 → P2 (primeira linha), depois P3 → P4 (segunda linha)")

        # Aguarda até 4 cliques
        while True:
            cv2.imshow("calibragem", clone)
            key = cv2.waitKey(1) & 0xFF
            if len(self.court_points) == 4:
                break
            if key == 27:  # ESC para sair
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        # Calcula distâncias em pixels
        dist1_px = math.hypot(self.court_points[1][0] - self.court_points[0][0],
                            self.court_points[1][1] - self.court_points[0][1])
        dist2_px = math.hypot(self.court_points[3][0] - self.court_points[2][0],
                            self.court_points[3][1] - self.court_points[2][1])

        # Média
        dist_media_px = (dist1_px + dist2_px) / 2
        pixels_por_metro = dist_media_px / 6.0  # 6 metros reais

        self.log.debug(f"Distância 1: {dist1_px:.2f} px")
        self.log.debug(f"Distância 2: {dist2_px:.2f} px")
        self.log.debug(f"Pixels por metro: {pixels_por_metro:.4f}")

        return pixels_por_metro

    def _exit(self) -> None:
        self.log.info('Saindo do módulo de processamento.')
        self.cap.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()