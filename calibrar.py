"""
Script de Calibração de Câmera — Tabuleiro de Xadrez
Uso:
  1. Imprima um tabuleiro de xadrez (9x6 cantos internos).
  2. Rode este script: python calibrar.py
  3. Mostre o tabuleiro para a câmera em diferentes ângulos e distâncias.
  4. Pressione [S] para salvar cada captura (mínimo 10, ideal 15-20).
  5. Pressione [C] para calcular a calibração e salvar calibracao.npz.
  6. Pressione [Q] para sair.
"""

import cv2
import numpy as np
from pathlib import Path

CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 25.0  # Tamanho real de cada quadrado do tabuleiro impresso (em mm)
CAMERA_INDEX = 0
OUTPUT_FILE = "calibracao.npz"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Pontos 3D do mundo real (Z=0 pois o tabuleiro é plano)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
imgpoints = []
capture_count = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Erro: não foi possível abrir a câmera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("=== CALIBRAÇÃO DE CÂMERA ===")
print(f"Tabuleiro: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} cantos internos")
print(f"Tamanho do quadrado: {SQUARE_SIZE_MM} mm")
print("[S] Capturar  |  [C] Calibrar  |  [Q] Sair")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners_refined, found)
        cv2.putText(display, "Tabuleiro DETECTADO - Pressione [S] para capturar",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 100), 2)
    else:
        cv2.putText(display, "Procurando tabuleiro...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(display, f"Capturas: {capture_count}/15",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Calibracao - Tabuleiro de Xadrez", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") or key == ord("S"):
        if found:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            capture_count += 1
            print(f"  Captura #{capture_count} salva!")
        else:
            print("  Tabuleiro não detectado. Ajuste a posição.")

    elif key == ord("c") or key == ord("C"):
        if capture_count < 5:
            print(f"  Poucas capturas ({capture_count}). Mínimo recomendado: 10.")
        else:
            print(f"\nCalculando calibração com {capture_count} capturas...")
            ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            # Calcula o erro de reprojeção
            total_error = 0
            for i in range(len(objpoints)):
                projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
                total_error += error
            mean_error = total_error / len(objpoints)

            np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)

            print(f"\nCalibração salva em: {OUTPUT_FILE}")
            print(f"Erro médio de reprojeção: {mean_error:.4f} pixels")
            print(f"\nMatriz Intrínseca:\n{mtx}")
            print(f"\nCoeficientes de Distorção:\n{dist}")
            print("\nAgora basta rodar o main.py — ele carregará a calibração automaticamente.")
            break

    elif key == ord("q") or key == ord("Q"):
        print("Calibração cancelada.")
        break

cap.release()
cv2.destroyAllWindows()
