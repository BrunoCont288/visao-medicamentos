"""
Módulos de visão computacional do Apsen Vision System.

Contém:
- CameraManager: gerenciamento de câmera com calibração de lente
- Detector: inferência YOLOv8 para localizar caixas de medicamento
- TextRecognizer: EasyOCR com suporte a rotações e fuzzy matching
- VisionPipeline: pipeline multithread que integra YOLO + OCR
"""

import time
import threading
from dataclasses import replace
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

from config import CONFIG, Detection, get_logger
from database import MedicineDatabase


logger = get_logger("Vision")



class CameraManager:

    def __init__(self, camera_index: int = 0, width: int = 1920,
                 height: int = 1080) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._mtx: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._new_mtx: Optional[np.ndarray] = None
        self._load_calibration()
        logger.info(f"Inicializando câmera {camera_index} ({width}x{height})...")

    def _load_calibration(self) -> None:
        cal_path = Path(CONFIG["calibration_path"])
        if cal_path.exists():
            data = np.load(str(cal_path))
            self._mtx = data["mtx"]
            self._dist = data["dist"]
            # new_mtx será calculada após conhecer a resolução real — vide open()
            logger.info(f"Calibração de lente carregada: {cal_path}")
        else:
            logger.info("Arquivo de calibração não encontrado. Sem correção de distorção.")

    def open(self) -> bool:
        """Abre a câmera e força a resolução Full HD."""
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback sem DSHOW
            self._cap = cv2.VideoCapture(self._index)

        if not self._cap.isOpened():
            logger.error(f"Falha ao abrir câmera {self._index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Libera o FPS da câmera USB
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        real_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calcular newCameraMatrix otimizada após saber a resolução real
        if self._mtx is not None:
            self._new_mtx, _ = cv2.getOptimalNewCameraMatrix(
                self._mtx, self._dist, (real_w, real_h), 1, (real_w, real_h)
            )
            logger.info("Matriz de câmera otimizada calculada.")

        logger.info(f"Camera aberta: {real_w}x{real_h}")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if ret and frame is not None and self._mtx is not None:
            new_mtx = self._new_mtx if self._new_mtx is not None else self._mtx
            frame = cv2.undistort(frame, self._mtx, self._dist, None, new_mtx)
        return ret, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera liberada.")

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap is None:
            return 0, 0
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))



class Detector:

    def __init__(self, model_path: str, confidence: float = 0.4) -> None:
        self._model_path = Path(model_path)
        self._confidence = confidence
        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        if not self._model_path.exists():
            logger.error(f"Modelo YOLO não encontrado: {self._model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {self._model_path}")

        logger.info(f"Carregando modelo YOLO: {self._model_path}")
        self._model = YOLO(str(self._model_path))
        logger.info("Modelo YOLO carregado com sucesso.")

    def detect(self, frame: np.ndarray) -> List[Tuple[List[int], float]]:
        """Executa inferência YOLO e retorna bounding boxes com confiança.

        Args:
            frame: Imagem BGR do OpenCV.

        Returns:
            Lista de tuplas [([x1, y1, x2, y2], confiança), ...].
        """
        if self._model is None:
            return []

        results = self._model(frame, conf=self._confidence, verbose=False)
        boxes: List[Tuple[List[int], float]] = []

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                boxes.append(([x1, y1, x2, y2], conf))

        return boxes



class TextRecognizer:

    def __init__(self, languages: List[str], gpu: bool = False) -> None:
        logger.info(f"Inicializando EasyOCR (langs={languages}, gpu={gpu})...")
        self._reader = easyocr.Reader(languages, gpu=gpu)
        logger.info("EasyOCR pronto.")

    def extract_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Extrai textos da imagem com suas respectivas confianças.

        Args:
            image: ROI recortada (BGR).

        Returns:
            Lista de tuplas [(texto, confiança), ...].
        """
        try:
            results = self._reader.readtext(image)
            extracted: List[Tuple[str, float]] = []

            for (_, text, prob) in results:
                if prob >= CONFIG["ocr_min_confidence"] and len(text.strip()) >= 3:
                    extracted.append((text.strip(), prob))

            return extracted
        except Exception as e:
            logger.warning(f"Erro no OCR: {e}")
            return []

    def identify(self, image: np.ndarray,
                 database: MedicineDatabase, last_rot: str = "0") -> Tuple[Optional[Detection], str, str]:
        """Executa OCR na ROI e faz fuzzy matching com o banco de dados em tempo real."""
        rotations = {
            "0": image,
            "180": cv2.rotate(image, cv2.ROTATE_180),
            "90": cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            "270": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        }

        # Tenta a última rotação que deu certo primeiro para evitar 4 cálculos
        order = [last_rot]
        for r in rotations.keys():
            if r not in order:
                order.append(r)

        # Coleta textos durante o loop para evitar chamada OCR redundante
        all_seen_texts: List[str] = []

        for rot_label in order:
            img = rotations[rot_label]
            texts = self.extract_text(img)
            if not texts:
                continue

            # Acumula textos lidos para debug (sem precisar chamar OCR de novo)
            for text, _ in texts:
                if text not in all_seen_texts:
                    all_seen_texts.append(text)

            for text, ocr_conf in texts:
                nome, score = database.fuzzy_match(text)
                if nome is not None:
                    suffix = f" [{rot_label}°]" if rot_label != "0" else ""
                    debug = f"{nome} ({score}%){suffix}"
                    det = Detection(
                        nome=nome,
                        confianca_ocr=ocr_conf,
                        similaridade_fuzzy=score,
                        status="IDENTIFICADO",
                        debug_ocr=debug,
                    )
                    return det, debug, rot_label

        # Nenhum match — reutiliza textos já coletados (sem chamada OCR extra)
        debug = f"OCR: {' | '.join(all_seen_texts[:3])}" if all_seen_texts else "Sem leitura"
        return None, debug, "0"



class VisionPipeline:
    def __init__(self) -> None:
        self._database = MedicineDatabase(CONFIG["database_path"])
        self._detector = Detector(CONFIG["model_path"], CONFIG["yolo_confidence"])
        self._recognizer = TextRecognizer(
            CONFIG["ocr_languages"], CONFIG["gpu_enabled"]
        )

        self._lock = threading.Lock()
        self._current_detection: Optional[Detection] = None
        self._current_boxes: List[List[int]] = []
        self._debug_text: str = "Aguardando..."
        self._last_identified_time: float = time.time()
        self._last_successful_rotation: str = "0"

        self._task_queue: Queue = Queue(maxsize=1)
        self._running: bool = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        logger.info("VisionPipeline inicializado.")

    def _worker_loop(self) -> None:
        """Thread que processa OCR sem bloquear o loop principal."""
        while self._running:
            try:
                frame, boxes_with_conf = self._task_queue.get(timeout=0.5)
            except Empty:
                continue

            if not boxes_with_conf:
                continue

            h, w = frame.shape[:2]
            margin = CONFIG["crop_margin_px"]

            for box, yolo_conf in boxes_with_conf:
                x1, y1, x2, y2 = box
                # Crop com margem segura
                cx1 = max(0, x1 - margin)
                cy1 = max(0, y1 - margin)
                cx2 = min(w, x2 + margin)
                cy2 = min(h, y2 + margin)
                roi = frame[cy1:cy2, cx1:cx2]

                if roi.size == 0:
                    continue

                # Calcular contorno rotacionado (OBB) na ROI original
                rotated_rect = self._compute_rotated_rect(frame, box)

                # Redimensionar para o target configurado
                target_w = CONFIG["crop_target_width"]
                roi_h, roi_w = roi.shape[:2]
                if roi_w > target_w:
                    scale = target_w / roi_w
                    roi = cv2.resize(roi, (target_w, int(roi_h * scale)))

                # Pré-processamento: CLAHE para melhorar contraste em textos claros
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                roi = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

                det, debug, best_rot = self._recognizer.identify(
                    roi, self._database, self._last_successful_rotation
                )

                with self._lock:
                    if det is not None:
                        det.bbox = box
                        det.confianca_yolo = yolo_conf
                        det.rotated_rect = rotated_rect
                        self._current_detection = det
                        self._debug_text = debug
                        self._last_identified_time = time.time()
                        self._last_successful_rotation = best_rot
                    else:
                        self._debug_text = debug

    def submit_frame(self, frame: np.ndarray, boxes_with_conf: List[Tuple[List[int], float]]) -> None:
        """Envia um frame para processamento OCR (non-blocking)."""
        if self._task_queue.full():
            try:
                self._task_queue.get_nowait()
            except Empty:
                pass
        self._task_queue.put((frame.copy(), boxes_with_conf))

    @staticmethod
    def _compute_rotated_rect(frame: np.ndarray, box: List[int]) -> Optional[tuple]:
        """Calcula o retângulo rotacionado (OBB) da caixa de remédio.

        Usa segmentação por contraste para encontrar a borda da caixa
        contra o fundo da mesa (branco), ignorando detalhes internos
        como textos, logos e desenhos.

        Returns:
            Tupla (center, size, angle) compatível com cv2.boxPoints,
            ou None se não encontrar contorno válido.
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # 1. Expandir ROI além do YOLO bbox para capturar a transição caixa→mesa
        expand = 40
        rx1 = max(0, x1 - expand)
        ry1 = max(0, y1 - expand)
        rx2 = min(w, x2 + expand)
        ry2 = min(h, y2 + expand)
        roi = frame[ry1:ry2, rx1:rx2]

        if roi.size == 0:
            return None

        # 2. Blur PESADO (21x21) para apagar texto, logos e desenhos internos
        blurred = cv2.GaussianBlur(roi, (21, 21), 0)

        # 3. Converter para escala de cinza
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 4. Otsu: encontra automaticamente o limiar entre mesa branca e caixa colorida
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 5. Morfologia: fechar buracos internos e suavizar bordas
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        # 6. Contornos EXTERNOS apenas (ignora furos internos)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 7. Maior contorno = silhueta da caixa
        largest = max(contours, key=cv2.contourArea)

        # Validar: área mínima (pelo menos 15% do ROI)
        roi_area = roi.shape[0] * roi.shape[1]
        if cv2.contourArea(largest) < roi_area * 0.15:
            return None

        rect = cv2.minAreaRect(largest)

        # 8. Validar tamanho: o retângulo rotacionado deve ter tamanho
        #    compatível com o bbox do YOLO (tolerância de 50%)
        yolo_w = x2 - x1
        yolo_h = y2 - y1
        rect_long = max(rect[1][0], rect[1][1])
        rect_short = min(rect[1][0], rect[1][1])
        expected_long = max(yolo_w, yolo_h)
        expected_short = min(yolo_w, yolo_h)

        if expected_long > 0 and expected_short > 0:
            if rect_long > expected_long * 1.6 or rect_short > expected_short * 1.6:
                return None
            if rect_long < expected_long * 0.3 or rect_short < expected_short * 0.3:
                return None

        # Ajustar coordenadas do ROI de volta ao frame completo
        center = (rect[0][0] + rx1, rect[0][1] + ry1)
        return (center, rect[1], rect[2])

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """Calcula IoU (Intersection over Union) entre duas bounding boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update_boxes(self, boxes: List[List[int]]) -> None:
        """Atualiza as bounding boxes do YOLO (chamado a cada detecção)."""
        with self._lock:
            self._current_boxes = boxes
            if boxes and self._current_detection:
                # Encontrar a box mais próxima da detecção atual por IoU
                best_box = boxes[0]
                best_iou = 0.0
                for b in boxes:
                    iou = self._compute_iou(self._current_detection.bbox, b)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = b
                self._current_detection.bbox = best_box

    def clear_if_timeout(self) -> None:
        """Limpa a detecção verde se o OCR não validar o texto por N segundos."""
        with self._lock:
            elapsed = time.time() - self._last_identified_time
            if elapsed > CONFIG["detection_timeout_s"]:
                self._current_detection = None
                self._debug_text = "Aguardando..."
                self._last_successful_rotation = "0"

    @property
    def detection(self) -> Optional[Detection]:
        """Retorna uma cópia thread-safe da detecção atual."""
        with self._lock:
            if self._current_detection is None:
                return None
            return replace(self._current_detection)

    @property
    def boxes(self) -> List[List[int]]:
        with self._lock:
            return list(self._current_boxes)

    @property
    def debug_text(self) -> str:
        with self._lock:
            return self._debug_text

    @property
    def detector(self) -> Detector:
        """Acesso público ao detector YOLO (usado pelo loop principal)."""
        return self._detector

    def shutdown(self) -> None:
        """Encerra a thread worker."""
        self._running = False
        self._worker.join(timeout=2.0)
        logger.info("VisionPipeline encerrado.")
