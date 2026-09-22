"""
Renderização da interface visual (HUD) do Apsen Vision System.

Desenha bounding boxes (com suavização temporal), contornos rotacionados,
painel de informações (dashboard industrial), barra inferior de debug
e indicador de FPS sobre o frame de saída.

Melhorias v3.1:
- Suavização Temporal (EMA) para eliminar tremor nos contornos
- Contorno Rotacionado (OBB) via cv2.minAreaRect
- Cantoneiras de mira (corner brackets) estilo industrial
- Dois estados visuais: Amarelo (YOLO detectou) → Verde (OCR confirmou)
"""

from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

from config import CONFIG, Detection


class OverlayRenderer:
    """Renderiza a interface visual sobre o frame de saída.

    Agora é uma classe com estado para manter a suavização temporal
    das bounding boxes entre frames consecutivos.
    """

    # Cores (BGR)
    GREEN = (0, 220, 100)
    YELLOW = (0, 200, 255)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    DARK_BG = (30, 30, 30)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self) -> None:
        self._alpha: float = CONFIG["smoothing_alpha"]
        self._corner_len: int = CONFIG["contour_corner_length"]
        # Estado de suavização: guarda a última posição suavizada
        # para bounding boxes (AABB) e retângulos rotacionados
        self._smooth_boxes: Dict[int, List[float]] = {}  # idx -> [x1,y1,x2,y2] suavizado
        self._smooth_rotated: Optional[Tuple] = None  # (center, size, angle) suavizado

    # ------------------------------------------------------------------
    # Suavização Temporal (EMA)
    # ------------------------------------------------------------------

    def _smooth_aabb(self, idx: int, box: List[int]) -> List[int]:
        """Aplica EMA (Exponential Moving Average) a uma bounding box AABB."""
        alpha = self._alpha
        new = [float(v) for v in box]

        if idx in self._smooth_boxes:
            prev = self._smooth_boxes[idx]
            smoothed = [alpha * n + (1 - alpha) * p for n, p in zip(new, prev)]
        else:
            smoothed = new

        self._smooth_boxes[idx] = smoothed
        return [int(round(v)) for v in smoothed]

    def _smooth_rotated_rect(self, rect: Tuple) -> Tuple:
        """Aplica EMA ao retângulo rotacionado (center, size, angle)."""
        alpha = self._alpha
        center, size, angle = rect

        if self._smooth_rotated is not None:
            pc, ps, pa = self._smooth_rotated
            # Suavizar centro
            sc = (alpha * center[0] + (1 - alpha) * pc[0],
                  alpha * center[1] + (1 - alpha) * pc[1])
            # Suavizar tamanho
            ss = (alpha * size[0] + (1 - alpha) * ps[0],
                  alpha * size[1] + (1 - alpha) * ps[1])
            # Suavizar ângulo (cuidado com a descontinuidade em ±90°)
            diff = angle - pa
            if diff > 45:
                diff -= 90
            elif diff < -45:
                diff += 90
            sa = pa + alpha * diff
        else:
            sc, ss, sa = center, size, angle

        self._smooth_rotated = (sc, ss, sa)
        return (sc, ss, sa)

    # ------------------------------------------------------------------
    # Desenho de Cantoneiras (Corner Brackets)
    # ------------------------------------------------------------------

    def _draw_corner_brackets(self, frame: np.ndarray, points: np.ndarray,
                               color: Tuple, thickness: int = 2) -> None:
        """Desenha cantoneiras nos 4 vértices de um polígono de 4 pontos."""
        corner_len = self._corner_len
        pts = [tuple(p) for p in points.astype(int)]

        for i in range(4):
            p1 = np.array(pts[i], dtype=float)
            p_prev = np.array(pts[(i - 1) % 4], dtype=float)
            p_next = np.array(pts[(i + 1) % 4], dtype=float)

            # Vetor do vértice para o vizinho anterior
            v1 = p_prev - p1
            len1 = np.linalg.norm(v1)
            if len1 > 0:
                v1 = v1 / len1
                end1 = p1 + v1 * min(corner_len, len1 * 0.3)
                cv2.line(frame, tuple(p1.astype(int)), tuple(end1.astype(int)),
                         color, thickness, cv2.LINE_AA)

            # Vetor do vértice para o vizinho seguinte
            v2 = p_next - p1
            len2 = np.linalg.norm(v2)
            if len2 > 0:
                v2 = v2 / len2
                end2 = p1 + v2 * min(corner_len, len2 * 0.3)
                cv2.line(frame, tuple(p1.astype(int)), tuple(end2.astype(int)),
                         color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Desenho Principal
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, detection: Optional[Detection],
             boxes: List[List[int]], fps: float, debug: str) -> np.ndarray:
        """Desenha bounding boxes, labels e painel fixo de informações.

        Args:
            frame: Frame BGR do OpenCV.
            detection: Detecção atual (ou None se não identificado).
            boxes: Lista de bounding boxes do YOLO (podem estar sem OCR).
            fps: FPS atual do loop principal.
            debug: Texto de debug do OCR.
        """
        h, w = frame.shape[:2]

        # Limpar boxes suavizadas que não existem mais
        active_indices = set(range(len(boxes)))
        stale = [k for k in self._smooth_boxes if k not in active_indices]
        for k in stale:
            del self._smooth_boxes[k]

        # --- Desenhar TODAS as bounding boxes do YOLO ---
        identified_box = detection.bbox if detection and detection.status == "IDENTIFICADO" else None

        if identified_box is None or (detection and detection.rotated_rect is None):
            self._smooth_rotated = None

        for i, box in enumerate(boxes):
            smoothed = self._smooth_aabb(i, box)
            sx1, sy1, sx2, sy2 = smoothed

            is_identified = (
                identified_box is not None
                and self._boxes_overlap(box, identified_box)
            )

            if is_identified and detection is not None:
                # --- ESTADO VERDE: OCR confirmou o medicamento ---
                color = self.GREEN

                # Tentar desenhar contorno rotacionado
                if detection.rotated_rect is not None:
                    smooth_rect = self._smooth_rotated_rect(detection.rotated_rect)
                    box_points = cv2.boxPoints(smooth_rect).astype(np.intp)

                    # Contorno rotacionado semi-transparente
                    overlay = frame.copy()
                    cv2.drawContours(overlay, [box_points], 0, color, 2, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

                    # Cantoneiras nos vértices
                    self._draw_corner_brackets(frame, box_points.astype(float), color, 3)

                    # Label com nome do medicamento no topo do contorno rotacionado
                    top_y = int(min(box_points[:, 1]))
                    top_x = int(min(box_points[:, 0]))
                    self._draw_label(frame, detection.nome, top_x, top_y - 10, color)
                else:
                    # Fallback: retângulo AABB com cantoneiras
                    rect_pts = np.array([
                        [sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]
                    ], dtype=float)
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2, cv2.LINE_AA)
                    self._draw_corner_brackets(frame, rect_pts, color, 3)
                    self._draw_label(frame, detection.nome, sx1, sy1 - 10, color)
            else:
                # --- ESTADO AMARELO: YOLO detectou, OCR ainda processando ---
                color = self.YELLOW
                rect_pts = np.array([
                    [sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]
                ], dtype=float)

                # Retângulo amarelo tracejado (pontilhado)
                self._draw_dashed_rect(frame, sx1, sy1, sx2, sy2, color, 2)
                self._draw_corner_brackets(frame, rect_pts, color, 2)
                self._draw_label(frame, "Identificando...", sx1, sy1 - 10, color)

        # --- Painel Fixo de Informações (Dashboard Industrial) ---
        panel_w = 420
        panel_h = 160
        margin = 20

        # Fundo do painel semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin), (margin + panel_w, margin + panel_h), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Borda do painel
        cv2.rectangle(frame, (margin, margin), (margin + panel_w, margin + panel_h), self.GREEN, 1)

        # Cabeçalho do painel
        cv2.putText(frame, "APSEN VISION SYSTEM", (margin + 15, margin + 30), self.FONT, 0.7, self.WHITE, 2)
        cv2.line(frame, (margin, margin + 40), (margin + panel_w, margin + 40), self.GREEN, 1)

        # Informações Dinâmicas
        if detection and detection.status == "IDENTIFICADO":
            # Medicamento
            cv2.putText(frame, "MEDICAMENTO:", (margin + 15, margin + 75), self.FONT, 0.6, self.WHITE, 1)
            cv2.putText(frame, detection.nome, (margin + 15, margin + 105), self.FONT, 0.9, self.GREEN, 2)

            # Confiança (Porcentagem Fuzzy)
            conf_text = f"Confianca: {detection.similaridade_fuzzy}%"
            cv2.putText(frame, conf_text, (margin + 15, margin + 140), self.FONT, 0.6, self.WHITE, 1)
        else:
            cv2.putText(frame, "MEDICAMENTO:", (margin + 15, margin + 75), self.FONT, 0.6, self.WHITE, 1)
            cv2.putText(frame, "Buscando...", (margin + 15, margin + 105), self.FONT, 0.9, self.YELLOW, 2)
            cv2.putText(frame, "Confianca: --", (margin + 15, margin + 140), self.FONT, 0.6, self.WHITE, 1)

        # FPS (Canto inferior direito do painel)
        fps_color = self.GREEN if fps >= 20 else self.YELLOW if fps >= 10 else self.RED
        cv2.putText(frame, f"FPS: {fps:.0f}", (margin + panel_w - 90, margin + 140), self.FONT, 0.6, fps_color, 2)

        # --- Barra Inferior (Debug e Atalhos) ---
        cv2.rectangle(frame, (0, h - 30), (w, h), self.DARK_BG, -1)
        status_text = f"Leitura OCR em tempo real: {debug}  |  Pressione [Q] para fechar"
        cv2.putText(frame, status_text, (15, h - 10), self.FONT, 0.5, self.WHITE, 1)

        return frame

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _draw_label(self, frame: np.ndarray, text: str,
                    x: int, y: int, color: Tuple) -> None:
        """Desenha um label com fundo semi-transparente."""
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, self.FONT, font_scale, thickness)

        # Fundo do label
        pad = 4
        lx1 = max(0, x - pad)
        ly1 = max(0, y - th - pad)
        lx2 = x + tw + pad
        ly2 = y + pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), self.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, text, (x, y), self.FONT, font_scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_rect(frame: np.ndarray, x1: int, y1: int,
                           x2: int, y2: int, color: Tuple,
                           thickness: int = 2, dash_length: int = 10) -> None:
        """Desenha um retângulo tracejado (pontilhado)."""
        edges = [
            ((x1, y1), (x2, y1)),  # Topo
            ((x2, y1), (x2, y2)),  # Direita
            ((x2, y2), (x1, y2)),  # Base
            ((x1, y2), (x1, y1)),  # Esquerda
        ]
        for (px1, py1), (px2, py2) in edges:
            dist = int(np.hypot(px2 - px1, py2 - py1))
            if dist == 0:
                continue
            dx = (px2 - px1) / dist
            dy = (py2 - py1) / dist
            for i in range(0, dist, dash_length * 2):
                start = (int(px1 + dx * i), int(py1 + dy * i))
                end_i = min(i + dash_length, dist)
                end = (int(px1 + dx * end_i), int(py1 + dy * end_i))
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _boxes_overlap(box_a: List[int], box_b: List[int], threshold: float = 0.5) -> bool:
        """Verifica se duas bounding boxes se sobrepõem (IoU simplificado)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        min_area = min(area_a, area_b) if min(area_a, area_b) > 0 else 1

        return (inter_area / min_area) >= threshold
