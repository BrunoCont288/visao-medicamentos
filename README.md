# 🏭 Apsen Vision System v3.0

![Apsen Farmacêutica](https://img.shields.io/badge/Apsen-Industrial_Vision-00b140?style=for-the-badge)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-FFD700?style=for-the-badge)
![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch_CUDA-EE4C2C?style=for-the-badge&logo=pytorch)

Sistema de Visão Computacional de alta performance desenvolvido para a **linha de embalagem industrial da Apsen Farmacêutica**. O sistema identifica medicamentos em tempo real cruzando inteligência artificial visual com leitura semântica de textos.

## 🚀 Arquitetura do Sistema

O pipeline de identificação trabalha de forma multithread para garantir máximo de FPS sem gargalos:

1. **YOLOv8 (Detecção Espacial)**: Redes neurais localizam as caixas em qualquer rotação ou posição na esteira e geram *Bounding Boxes*.
2. **Crop & Otimização**: A imagem (ROI) é recortada com precisão cirúrgica, otimizada para 320px e tratada com CLAHE (contraste adaptativo) para facilitar a leitura sem afogar a GPU.
3. **EasyOCR CUDA (Leitura de Dados)**: Um processamento na GPU extrai os textos impressos na caixa em segundo plano, testando rotações de 0º, 90º, 180º e 270º.
4. **Fuzzy Matching**: Algoritmos de correlação de strings compensam ruídos na câmera (ex: "D1p1r0na") e cruzam a leitura com o banco de dados oficial (SKUs).

## 📂 Estrutura de Arquivos e Módulos (.py)

O sistema é 100% modularizado, separando configurações, processamento de IA, interface e banco de dados:

| Arquivo | Responsabilidade | O que faz no sistema |
| :--- | :--- | :--- |
| [`main.py`](file:///c:/Users/Bruno%20Cont/Desktop/Faculdade/Visão%20computacional/main.py) | **Orquestrador / Loop Principal** | Ponto de entrada (`entry point`). Inicializa a câmera, roda o loop em tempo real respeitando a taxa de FPS, coordena o envio de frames para a IA, gerencia comandos de teclado (`Q` para fechar) e finaliza threads com segurança. |
| [`config.py`](file:///c:/Users/Bruno%20Cont/Desktop/Faculdade/Visão%20computacional/config.py) | **Configurações & Estruturas** | Centraliza todas as constantes e hiperparâmetros no dicionário `CONFIG` (índice da câmera, resolução, thresholds, suavização EMA), padroniza o sistema de `logging` e define o dataclass `Detection`. |
| [`vision.py`](file:///c:/Users/Bruno%20Cont/Desktop/Faculdade/Visão%20computacional/vision.py) | **Pipeline de Visão & IA** | Contém o núcleo de Visão Computacional:<br>• `CameraManager`: captura de vídeo e desdistorção óptica da lente.<br>• `Detector`: inferência do modelo YOLOv8 (`best.pt`).<br>• `TextRecognizer`: extração de caracteres com EasyOCR em 4 rotações.<br>• `VisionPipeline`: orquestrador multithread assíncrono com pré-processamento CLAHE e cálculo do contorno rotacionado (`cv2.minAreaRect`). |
| [`overlay.py`](file:///c:/Users/Bruno%20Cont/Desktop/Faculdade/Visão%20computacional/overlay.py) | **Interface Visual (HUD)** | Desenha a camada visual industrial sobre o vídeo:<br>• **Suavização Temporal (EMA)**: elimina tremores (*jitter*) nos contornos.<br>• **Feedback em 2 Estados**: contorno amarelo tracejado ("Identificando...") e verde rotacionado com cantoneiras de mira ("Remédio Confirmado").<br>• Dashboard semi-transparente com status, métricas de confiança e FPS. |
| [`database.py`](file:///c:/Users/Bruno%20Cont/Desktop/Faculdade/Visão%20computacional/database.py) | **Catálogo & Fuzzy Matching** | Lê o banco de dados oficial (`medicamentos_apsen.json`) e aplica o algoritmo de busca flexível (`thefuzz`) para identificar o medicamento mesmo quando a câmera sofre com reflexos ou letras borradas. |
| [`calibrar.py`](file:///c:/Users/Bruno%20Cont/Desktop/Faculdade/Visão%20computacional/calibrar.py) | **Calibração de Câmera** | Script utilitário independente para calibrar a câmera usando padrão de xadrez (*checkerboard*). Gera a matriz intrínseca e coeficientes de distorção em `calibracao.npz` para eliminar distorções de lente. |

## 📦 Medicamentos Suportados (SKUs)
O sistema atualmente audita o empacotamento de 4 famílias principais (conforme `medicamentos_apsen.json`):
- **Alois (Apsen)** - `5mg`, `10mg`, `Duo`
- **Flancox (Apsen)** - `300mg`, `400mg`, `500mg`
- **Miosan (Apsen)** - `5mg`, `10mg`, `Caf`, `ODT`
- **Dipirona (Genérico)** - `500mg`, `Sódica`

## ⚙️ Requisitos de Hardware

Para atingir leitura instantânea sem *flickering* (piscar), o sistema foi programado para usar o poder paralelo das placas NVIDIA. 
* **Processador**: Intel Core i5 / Ryzen 5 (ou superior)
* **Vídeo**: NVIDIA MX350, GTX, RTX, ou Tesla T4 (Obrigatório suporte a CUDA 11.8+)
* **Câmera**: Webcam Full HD 1080p (com suporte a MJPG)

## 🛠️ Instalação e Execução

### 1. Instalar o PyTorch com suporte a Placa de Vídeo (CUDA)
Para o sistema funcionar na velocidade ideal, baixe a versão de aceleração CUDA:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Instalar Dependências Restantes
```bash
pip install opencv-python ultralytics easyocr thefuzz
```

### 3. Rodar a Aplicação
```bash
python main.py
```