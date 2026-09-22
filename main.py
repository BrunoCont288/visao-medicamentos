"""
Apsen Vision System v3.0 — Loop Principal.

Ponto de entrada da aplicação. Integra câmera, pipeline de visão
(YOLO + OCR) e renderização do HUD. Toda a lógica de cada componente
está nos seus respectivos módulos:

- config.py   → Configurações, logging e dataclass Detection
- database.py → MedicineDatabase e fuzzy matching
- vision.py   → CameraManager, Detector, TextRecognizer, VisionPipeline
- overlay.py  → OverlayRenderer (HUD industrial)
"""

import time
from typing import List

import cv2

from config import CONFIG, get_logger
from vision import CameraManager, VisionPipeline
from overlay import OverlayRenderer


logger = get_logger("Main")



def main() -> None:
    logger.info("=" * 60)
    logger.info("  APSEN VISION SYSTEM v3.0 — Iniciando...")
    logger.info("=" * 60)

    # Inicialização
    camera = CameraManager(
        CONFIG["camera_index"],
        CONFIG["camera_width"],
        CONFIG["camera_height"],
    )

    if not camera.open():
        logger.critical("Não foi possível abrir a câmera. Encerrando.")
        return

    pipeline = VisionPipeline()
    renderer = OverlayRenderer()  # Instância com estado (suavização temporal)

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

            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Frame vazio, tentando novamente...")
                continue

            frame_count += 1

            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = curr_time

            if frame_count % CONFIG["yolo_skip_frames"] == 0:
                current_boxes = pipeline.detector.detect(frame)
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

            pipeline.clear_if_timeout()

            det = pipeline.detection
            boxes = last_yolo_boxes
            debug = pipeline.debug_text

            frame = renderer.draw(frame, det, boxes, fps, debug)

            cv2.imshow("APSEN VISION SYSTEM v3.0", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Tecla [Q] pressionada. Encerrando...")
                break

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


if __name__ == "__main__":
    main()
