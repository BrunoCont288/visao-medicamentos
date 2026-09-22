"""
Configurações centrais do Apsen Vision System v3.0.

Contém todas as constantes, parâmetros de configuração,
setup de logging e o dataclass Detection.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


CONFIG: Dict = {
    "model_path": "best.pt",
    "database_path": "medicamentos_apsen.json",
    "camera_index": 0,  # Índice da câmera (0 = padrão, 1 = webcam externa)
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
    "crop_target_width": 320,          # Largura alvo do crop (era 250, agora maior p/ melhor OCR)
    "smoothing_alpha": 0.35,            # Fator EMA para suavização temporal (menor = mais suave)
    "contour_corner_length": 20,        # Tamanho das cantoneiras do contorno rotacionado (px)
    "gpu_enabled": True,
    "detection_timeout_s": 1.0,
    "calibration_path": "calibracao.npz",
}


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado com o padrão do sistema."""
    return logging.getLogger(name)


# --- Dataclass de Detecção ---
@dataclass
class Detection:
    nome: str = "Desconhecido"
    confianca_yolo: float = 0.0
    confianca_ocr: float = 0.0
    similaridade_fuzzy: int = 0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    rotated_rect: Optional[Tuple] = None  # (center, size, angle) do cv2.minAreaRect
    status: str = "BUSCANDO"
    debug_ocr: str = ""
    timestamp: float = field(default_factory=time.time)
