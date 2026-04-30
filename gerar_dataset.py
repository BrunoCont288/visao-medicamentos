"""
==============================================================================
  GERADOR DE DATASET SINTETICO v2.2 - Apsen Vision System
==============================================================================
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import Tuple, List

# --- CONFIGURACAO ---
CONFIG = {
    "input_dir": "imagens_base",
    "output_dir": "dataset_sintetico",
    "images_per_medicine": 500,
    "output_size": 640,
    "min_scale": 0.30,
    "max_scale": 0.80,
}

# --- BACKGROUNDS ---
def bg_white(s: int) -> np.ndarray:
    v = random.randint(225, 255)
    bg = np.full((s, s, 3), v, dtype=np.uint8)
    noise = np.random.randint(0, 6, (s, s, 3), dtype=np.uint8)
    return cv2.add(bg, noise)

def bg_gray(s: int) -> np.ndarray:
    v = random.randint(130, 200)
    bg = np.full((s, s, 3), v, dtype=np.uint8)
    noise = np.random.randint(0, 10, (s, s, 3), dtype=np.uint8)
    return cv2.add(bg, noise)

def bg_dark(s: int) -> np.ndarray:
    v = random.randint(30, 90)
    bg = np.full((s, s, 3), v, dtype=np.uint8)
    noise = np.random.randint(0, 8, (s, s, 3), dtype=np.uint8)
    return cv2.add(bg, noise)

def bg_gradient(s: int) -> np.ndarray:
    bg = np.zeros((s, s, 3), dtype=np.uint8)
    c1 = np.array([random.randint(100, 255) for _ in range(3)])
    c2 = np.array([random.randint(100, 255) for _ in range(3)])
    for y in range(s):
        t = y / s
        bg[y, :] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    return bg

def bg_colored(s: int) -> np.ndarray:
    color = [random.randint(80, 220) for _ in range(3)]
    bg = np.full((s, s, 3), color, dtype=np.uint8)
    noise = np.random.randint(0, 12, (s, s, 3), dtype=np.uint8)
    return cv2.add(bg, noise)

def bg_wood(s: int) -> np.ndarray:
    base = random.randint(100, 160)
    bg = np.full((s, s, 3), [base, int(base * 0.85), int(base * 0.7)], dtype=np.uint8)
    for y in range(0, s, random.randint(8, 20)):
        stripe_color = [base + random.randint(-15, 15),
                        int(base * 0.85) + random.randint(-10, 10),
                        int(base * 0.7) + random.randint(-10, 10)]
        stripe_color = [max(0, min(255, c)) for c in stripe_color]
        cv2.line(bg, (0, y), (s, y), stripe_color, random.randint(1, 3))
    return cv2.add(bg, np.random.randint(0, 8, (s, s, 3), dtype=np.uint8))

def random_background(s: int) -> np.ndarray:
    return random.choices([bg_white, bg_gray, bg_dark, bg_gradient, bg_colored, bg_wood], 
                          weights=[0.3, 0.2, 0.1, 0.15, 0.1, 0.15], k=1)[0](s)

# --- AUGMENTACOES ---
def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotaciona a imagem BGRA preenchendo as bordas com alpha=0."""
    h, w = img.shape[:2]
    rad = math.radians(abs(angle))
    new_w = int(w * abs(math.cos(rad)) + h * abs(math.sin(rad)))
    new_h = int(h * abs(math.cos(rad)) + w * abs(math.sin(rad)))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def mild_perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = int(min(w, h) * 0.04)
    src = np.float32([[0,0], [w,0], [w,h], [0,h]])
    dst = np.float32([[random.randint(0,m), random.randint(0,m)],
                      [w-random.randint(0,m), random.randint(0,m)],
                      [w-random.randint(0,m), h-random.randint(0,m)],
                      [random.randint(0,m), h-random.randint(0,m)]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def adjust_brightness_contrast(img: np.ndarray) -> np.ndarray:
    alpha, beta = random.uniform(0.7, 1.4), random.randint(-30, 30)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

def apply_color_jitter(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-10,10)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.8, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_noise(img: np.ndarray) -> np.ndarray:
    sigma = random.uniform(3, 12)
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def apply_shadow(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    pts = np.array([[random.randint(0,w//2),0],[random.randint(w//2,w),0],[random.randint(w//2,w),h],[random.randint(0,w//2),h]], np.int32)
    cv2.fillPoly(overlay, [pts], (0,0,0))
    alpha = random.uniform(0.05, 0.2)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

# --- COMPOSICAO ---
def compose_image(medicine_bgra: np.ndarray, size: int) -> Tuple[np.ndarray, List[float]]:
    img = medicine_bgra.copy()
    
    # Transformacoes espaciais
    img = rotate_image(img, random.uniform(0,360) if random.random() < 0.15 else random.uniform(-30,30))
    if random.random() < 0.4: img = mild_perspective(img)
    
    # Transformacoes de cor (apenas nos canais BGR)
    if random.random() < 0.6:
        bgr = adjust_brightness_contrast(img[:,:,:3])
        img[:,:,:3] = bgr
    if random.random() < 0.3:
        bgr = apply_color_jitter(img[:,:,:3])
        img[:,:,:3] = bgr
    
    # Resize
    h, w = img.shape[:2]
    scale = random.uniform(CONFIG["min_scale"], CONFIG["max_scale"])
    tw = min(int(size * scale), size-4)
    th = min(int(tw * (h/max(w,1))), size-4)
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    
    # Prepara background
    bg = random_background(size)
    if random.random() < 0.3: bg = apply_shadow(bg)
    x, y = random.randint(0, max(0, size-tw)), random.randint(0, max(0, size-th))
    
    # Aplica o blend corretamente
    roi = bg[y:y+th, x:x+tw]
    alpha = np.expand_dims(img[:,:,3]/255.0, axis=2)
    bg[y:y+th, x:x+tw] = (alpha * img[:,:,:3] + (1-alpha) * roi).astype(np.uint8)
    
    # Efeitos finais
    if random.random() < 0.3: bg = apply_noise(bg)
    if random.random() < 0.15: bg = apply_blur(bg)
    
    # Bounding box considerando alpha channel
    # Encontra os limites reais da caixa baseados no canal alpha
    alpha_mask = img[:,:,3] > 10
    coords = cv2.findNonZero(alpha_mask.astype(np.uint8))
    if coords is not None:
        x_min, y_min, w_box, h_box = cv2.boundingRect(coords)
        real_cx = (x + x_min + w_box/2) / size
        real_cy = (y + y_min + h_box/2) / size
        real_w = w_box / size
        real_h = h_box / size
    else:
        # Fallback
        real_cx, real_cy = (x+tw/2)/size, (y+th/2)/size
        real_w, real_h = tw/size, th/size
        
    return bg, [real_cx, real_cy, real_w, real_h]

# --- MAIN ---
def main():
    in_dir, out_dir = Path(CONFIG["input_dir"]), Path(CONFIG["output_dir"])
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    if not in_dir.exists(): return print(f"Erro: {in_dir} nao existe")
    
    imgs = sorted([f for f in in_dir.iterdir() if f.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".webp"}])
    if not imgs: return print("Erro: Nenhuma imagem base")
    
    for d in [img_dir, lbl_dir]:
        if d.exists(): [f.unlink() for f in d.iterdir()]
        d.mkdir(parents=True, exist_ok=True)
        
    total = 0
    for idx, p in enumerate(imgs):
        name = p.stem
        
        # IMPORTANTE: Carrega mantendo o canal alpha (se for PNG transparente)
        base = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if base is None: continue
        
        # Garante formato BGRA
        if len(base.shape) == 2: # Gray
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGRA)
        elif base.shape[2] == 3: # BGR
            base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
            
        variants = [base, cv2.flip(base, 1)]
        for i in range(CONFIG["images_per_medicine"]):
            res, bbox = compose_image(random.choice(variants), CONFIG["output_size"])
            cv2.imwrite(str(img_dir / f"{name}_{i:04d}.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(lbl_dir / f"{name}_{i:04d}.txt", "w") as f:
                f.write(f"0 {' '.join(f'{v:.6f}' for v in bbox)}\n")
            total += 1
        print(f"[{idx+1}/{len(imgs)}] {name} OK")

    with open(out_dir / "data.yaml", "w") as f:
        f.write("path: ../dataset_sintetico\ntrain: images\nval: images\nnc: 1\nnames: ['caixa_remedio']\n")
    print(f"Pronto! {total} imagens geradas em {out_dir.absolute()}")

if __name__ == "__main__": main()
