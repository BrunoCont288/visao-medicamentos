"""
==============================================================================
  APSEN VISION SYSTEM v3.0
  Sistema de Visão Computacional em Tempo Real
  Identificação de Medicamentos — Apsen Farmacêutica
  Pipeline: YOLO Detection → ROI Crop → OCR → Fuzzy Matching (JSON)
==============================================================================
"""

# --- Standard Library ---
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Dict, Tuple

# --- Third Party ---
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from thefuzz import fuzz, process

# ==============================================================================
#  CONFIGURAÇÃO GLOBAL
# ==============================================================================

CONFIG: Dict = {
    "model_path": "best.pt",
    "database_path": "medicamentos_apsen.json",
    "camera_index": 1,  # Mudado para 1 (Webcam Externa)
    "camera_width": 1920,
    "camera_height": 1080,
    "target_fps": 30,
    "yolo_confidence": 0.5,  # Confiança média, mas a caixa só aparece após leitura do OCR
    "yolo_skip_frames": 3,
    "ocr_skip_frames": 5,
    "ocr_languages": ["pt", "en"],
    "ocr_min_confidence": 0.40,
    "fuzzy_threshold": 70,
    "crop_margin_px": 20,
    "gpu_enabled": True,
    "detection_timeout_s": 1.0,  # Estabilidade de 1 segundo (ideal para o ritmo da MX350)
}

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ApsenVision")


# ==============================================================================
#  DATACLASS: Detection
# ==============================================================================

@dataclass
class Detection:
    """Representa uma detecção de medicamento no frame."""
    nome: str = "Desconhecido"
    confianca_yolo: float = 0.0
    confianca_ocr: float = 0.0
    similaridade_fuzzy: int = 0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    status: str = "BUSCANDO"
    debug_ocr: str = ""
    timestamp: float = field(default_factory=time.time)


# ==============================================================================
#  CLASSE: MedicineDatabase
# ==============================================================================

class MedicineDatabase:
    """Gerencia o banco de dados JSON de medicamentos da Apsen."""

    def __init__(self, json_path: str) -> None:
        self._path = Path(json_path)
        self._data: List[Dict] = []
        self._all_names: List[str] = []
        self._name_to_official: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Carrega e indexa o JSON de medicamentos."""
        if not self._path.exists():
            logger.error(f"Arquivo não encontrado: {self._path}")
            raise FileNotFoundError(f"JSON não encontrado: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._data = raw.get("medicamentos", [])
        for med in self._data:
            nome_oficial = med.get("nome_oficial", "")
            for variante in med.get("variantes", []):
                self._all_names.append(variante)
                self._name_to_official[variante.upper()] = nome_oficial

        logger.info(f"Banco de dados carregado: {len(self._data)} medicamentos, "
                     f"{len(self._all_names)} variantes indexadas.")

    def fuzzy_match(self, ocr_text: str) -> Tuple[Optional[str], int]:
        """Compara texto OCR com variantes usando fuzzy matching.
        
        Returns:
            Tupla (nome_oficial, score) ou (None, 0) se abaixo do threshold.
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return None, 0

        cleaned = ocr_text.strip().upper()
        result = process.extractOne(
            cleaned,
            self._all_names,
            scorer=fuzz.token_sort_ratio
        )

        if result is None:
            return None, 0

        best_match, score, *_ = result

        if score >= CONFIG["fuzzy_threshold"]:
            official = self._name_to_official.get(best_match.upper(), best_match)
            return official, score

        return None, score

    @property
    def count(self) -> int:
        return len(self._data)


# ==============================================================================
#  CLASSE: CameraManager
# ==============================================================================

class CameraManager:
    """Gerencia a captura de vídeo da webcam com controle de resolução."""

    def __init__(self, camera_index: int = 0, width: int = 1920,
                 height: int = 1080) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        logger.info(f"Inicializando câmera {camera_index} ({width}x{height})...")

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
        logger.info(f"Camera aberta: {real_w}x{real_h}")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lê um frame da câmera."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        """Libera os recursos da câmera."""
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera liberada.")

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap is None:
            return 0, 0
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


# ==============================================================================
#  CLASSE: Detector (YOLO)
# ==============================================================================

class Detector:
    """Wrapper para o modelo YOLOv8 de detecção de objetos."""

    def __init__(self, model_path: str, confidence: float = 0.4) -> None:
        self._model_path = Path(model_path)
        self._confidence = confidence
        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Carrega os pesos do modelo YOLO."""
        if not self._model_path.exists():
            logger.error(f"Modelo YOLO não encontrado: {self._model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {self._model_path}")

        logger.info(f"Carregando modelo YOLO: {self._model_path}")
        self._model = YOLO(str(self._model_path))
        logger.info("Modelo YOLO carregado com sucesso.")

    def detect(self, frame: np.ndarray) -> List[List[int]]:
        """Executa inferência YOLO e retorna lista de bounding boxes [x1,y1,x2,y2].

        Args:
            frame: Imagem BGR do OpenCV.

        Returns:
            Lista de bounding boxes [[x1, y1, x2, y2], ...].
        """
        if self._model is None:
            return []

        results = self._model(frame, conf=self._confidence, verbose=False)
        boxes: List[List[int]] = []

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append([x1, y1, x2, y2])

        return boxes


# ==============================================================================
#  CLASSE: TextRecognizer (OCR + Fuzzy)
# ==============================================================================

class TextRecognizer:
    """Motor de OCR usando EasyOCR com filtro de confiança."""

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

        for rot_label in order:
            img = rotations[rot_label]
            texts = self.extract_text(img)
            if not texts:
                continue

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

        # Nenhum match encontrado
        all_texts = [t for t, _ in self.extract_text(rotations["0"])]
        debug = f"OCR: {' | '.join(all_texts[:3])}" if all_texts else "Sem leitura"
        return None, debug, "0"


# ==============================================================================
#  CLASSE: VisionPipeline (Orquestrador)
# ==============================================================================

class VisionPipeline:
    """Orquestra todo o pipeline de visão em uma thread separada.

    Fluxo: Frame → YOLO → Crop ROI → OCR (a cada N frames) → Fuzzy → Detection
    """

    def __init__(self) -> None:
        # --- Componentes ---
        self._database = MedicineDatabase(CONFIG["database_path"])
        self._detector = Detector(CONFIG["model_path"], CONFIG["yolo_confidence"])
        self._recognizer = TextRecognizer(
            CONFIG["ocr_languages"], CONFIG["gpu_enabled"]
        )

        # --- Estado compartilhado (thread-safe) ---
        self._lock = threading.Lock()
        self._current_detection: Optional[Detection] = None
        self._current_boxes: List[List[int]] = []
        self._debug_text: str = "Aguardando..."
        self._last_identified_time: float = time.time()
        self._last_successful_rotation: str = "0"

        # --- Worker thread ---
        self._task_queue: Queue = Queue(maxsize=1)
        self._running: bool = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        logger.info("VisionPipeline inicializado.")

    def _worker_loop(self) -> None:
        """Thread que processa OCR sem bloquear o loop principal."""
        while self._running:
            try:
                frame, boxes = self._task_queue.get(timeout=0.5)
            except Empty:
                continue

            if not boxes:
                continue

            h, w = frame.shape[:2]
            margin = CONFIG["crop_margin_px"]

            for box in boxes:
                x1, y1, x2, y2 = box
                # Crop com margem segura
                cx1 = max(0, x1 - margin)
                cy1 = max(0, y1 - margin)
                cx2 = min(w, x2 + margin)
                cy2 = min(h, y2 + margin)
                roi = frame[cy1:cy2, cx1:cx2]

                if roi.size == 0:
                    continue

                # OTIMIZAÇÃO EXTREMA PARA A MX350:
                # Reduzindo para 250 pixels. Metade do tamanho = OCR muito mais rápido
                roi_h, roi_w = roi.shape[:2]
                if roi_w > 250:
                    scale = 250 / roi_w
                    roi = cv2.resize(roi, (250, int(roi_h * scale)))

                det, debug, best_rot = self._recognizer.identify(
                    roi, self._database, self._last_successful_rotation
                )

                with self._lock:
                    if det is not None:
                        det.bbox = box
                        self._current_detection = det
                        self._debug_text = debug
                        self._last_identified_time = time.time()
                        self._last_successful_rotation = best_rot
                    else:
                        self._debug_text = debug

    def submit_frame(self, frame: np.ndarray, boxes: List[List[int]]) -> None:
        """Envia um frame para processamento OCR (non-blocking)."""
        if self._task_queue.full():
            try:
                self._task_queue.get_nowait()
            except Empty:
                pass
        self._task_queue.put((frame.copy(), boxes))

    def update_boxes(self, boxes: List[List[int]]) -> None:
        """Atualiza as bounding boxes do YOLO (chamado a cada detecção)."""
        with self._lock:
            self._current_boxes = boxes
            if boxes and self._current_detection:
                self._current_detection.bbox = boxes[0]

    def clear_if_timeout(self) -> None:
        """Limpa a detecção verde se o OCR não validar o texto por N segundos."""
        with self._lock:
            elapsed = time.time() - self._last_identified_time
            if elapsed > CONFIG["detection_timeout_s"]:
                self._current_detection = None
                self._debug_text = "Aguardando..."

    @property
    def detection(self) -> Optional[Detection]:
        with self._lock:
            return self._current_detection

    @property
    def boxes(self) -> List[List[int]]:
        with self._lock:
            return list(self._current_boxes)

    @property
    def debug_text(self) -> str:
        with self._lock:
            return self._debug_text

    def shutdown(self) -> None:
        """Encerra a thread worker."""
        self._running = False
        self._worker.join(timeout=2.0)
        logger.info("VisionPipeline encerrado.")


# ==============================================================================
#  CLASSE: OverlayRenderer (Interface Visual)
# ==============================================================================

class OverlayRenderer:
    """Renderiza a interface visual sobre o frame de saída."""

    # Cores (BGR)
    GREEN = (0, 220, 100)
    YELLOW = (0, 200, 255)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    DARK_BG = (30, 30, 30)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def draw(frame: np.ndarray, detection: Optional[Detection],
             boxes: List[List[int]], fps: float, debug: str) -> np.ndarray:
        """Desenha bounding boxes, labels e painel fixo de informações."""
        h, w = frame.shape[:2]

        # --- Bounding Boxes (Na caixa) ---
        if detection and detection.status == "IDENTIFICADO":
            x1, y1, x2, y2 = detection.bbox
            color = OverlayRenderer.GREEN

            # Retângulo verde em volta do remédio
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # Nome pequeno colado na caixa
            cv2.putText(frame, detection.nome, (x1, y1 - 10), OverlayRenderer.FONT, 0.6, color, 2)
        else:
            # Rastreador: Linha azul fina mostrando o que a câmera (YOLO) está focando
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)

        # --- Painel Fixo de Informações (Dashboard Industrial) ---
        panel_w = 420
        panel_h = 160
        margin = 20
        
        # Fundo do painel semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin), (margin + panel_w, margin + panel_h), OverlayRenderer.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Borda do painel
        cv2.rectangle(frame, (margin, margin), (margin + panel_w, margin + panel_h), OverlayRenderer.GREEN, 1)

        # Cabeçalho do painel
        cv2.putText(frame, "APSEN VISION SYSTEM", (margin + 15, margin + 30), OverlayRenderer.FONT, 0.7, OverlayRenderer.WHITE, 2)
        cv2.line(frame, (margin, margin + 40), (margin + panel_w, margin + 40), OverlayRenderer.GREEN, 1)

        # Informações Dinâmicas
        if detection and detection.status == "IDENTIFICADO":
            # Medicamento
            cv2.putText(frame, "MEDICAMENTO:", (margin + 15, margin + 75), OverlayRenderer.FONT, 0.6, OverlayRenderer.WHITE, 1)
            cv2.putText(frame, detection.nome, (margin + 15, margin + 105), OverlayRenderer.FONT, 0.9, OverlayRenderer.GREEN, 2)
            
            # Confiança (Porcentagem Fuzzy)
            conf_text = f"Confianca: {detection.similaridade_fuzzy}%"
            cv2.putText(frame, conf_text, (margin + 15, margin + 140), OverlayRenderer.FONT, 0.6, OverlayRenderer.WHITE, 1)
        else:
            cv2.putText(frame, "MEDICAMENTO:", (margin + 15, margin + 75), OverlayRenderer.FONT, 0.6, OverlayRenderer.WHITE, 1)
            cv2.putText(frame, "Buscando...", (margin + 15, margin + 105), OverlayRenderer.FONT, 0.9, OverlayRenderer.YELLOW, 2)
            cv2.putText(frame, "Confianca: --", (margin + 15, margin + 140), OverlayRenderer.FONT, 0.6, OverlayRenderer.WHITE, 1)

        # FPS (Canto inferior direito do painel)
        fps_color = OverlayRenderer.GREEN if fps >= 20 else OverlayRenderer.YELLOW if fps >= 10 else OverlayRenderer.RED
        cv2.putText(frame, f"FPS: {fps:.0f}", (margin + panel_w - 90, margin + 140), OverlayRenderer.FONT, 0.6, fps_color, 2)

        # --- Barra Inferior (Debug e Atalhos) ---
        cv2.rectangle(frame, (0, h - 30), (w, h), OverlayRenderer.DARK_BG, -1)
        status_text = f"Leitura OCR em tempo real: {debug}  |  Pressione [Q] para fechar"
        cv2.putText(frame, status_text, (15, h - 10), OverlayRenderer.FONT, 0.5, OverlayRenderer.WHITE, 1)

        return frame


# ==============================================================================
#  LOOP PRINCIPAL
# ==============================================================================

def main() -> None:
    """Ponto de entrada do sistema de visão computacional."""
    logger.info("=" * 60)
    logger.info("  APSEN VISION SYSTEM v3.0 — Iniciando...")
    logger.info("=" * 60)

    # --- Inicialização ---
    camera = CameraManager(
        CONFIG["camera_index"],
        CONFIG["camera_width"],
        CONFIG["camera_height"],
    )

    if not camera.open():
        logger.critical("Não foi possível abrir a câmera. Encerrando.")
        return

    pipeline = VisionPipeline()

    # --- Variáveis de controle ---
    frame_count: int = 0
    prev_time: float = time.time()
    fps: float = 0.0
    frame_interval: float = 1.0 / CONFIG["target_fps"]
    last_yolo_boxes: List[List[int]] = []
    last_detection_time: float = time.time()

    logger.info(f"Loop principal iniciado. Target: {CONFIG['target_fps']} FPS")
    logger.info("Pressione [Q] para sair.")

    try:
        while True:
            loop_start = time.time()

            # --- Captura ---
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Frame vazio, tentando novamente...")
                continue

            frame_count += 1

            # --- FPS ---
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = curr_time

            # --- YOLO: Detecção a cada N frames ---
            if frame_count % CONFIG["yolo_skip_frames"] == 0:
                current_boxes = pipeline._detector.detect(frame)
                if current_boxes:
                    last_yolo_boxes = current_boxes
                    last_detection_time = time.time()
                    pipeline.update_boxes(current_boxes)

                    # OCR: O pipeline tem uma fila de tamanho 1. Ele processa o mais rápido
                    # possível no plano de fundo. Submetemos o frame em toda detecção.
                    pipeline.submit_frame(frame, current_boxes)
                else:
                    # Timeout do YOLO: limpa boxes se não detectou nada físico por N segundos
                    if time.time() - last_detection_time > CONFIG["detection_timeout_s"]:
                        last_yolo_boxes = []

            # --- Limpeza de Memória do OCR ---
            # Sempre verifica se o OCR parou de ler o remédio há mais de N segundos
            pipeline.clear_if_timeout()

            # --- Renderização ---
            det = pipeline.detection
            boxes = last_yolo_boxes
            debug = pipeline.debug_text

            frame = OverlayRenderer.draw(frame, det, boxes, fps, debug)

            # --- Exibição ---
            cv2.imshow("APSEN VISION SYSTEM v3.0", frame)

            # --- Controle de teclas ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Tecla [Q] pressionada. Encerrando...")
                break

            # --- Throttle de FPS ---
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupção pelo teclado (Ctrl+C).")
    finally:
        pipeline.shutdown()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Sistema encerrado com sucesso.")


# ==============================================================================
if __name__ == "__main__":
    main()
