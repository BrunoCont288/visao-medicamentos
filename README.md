# 🏭 Apsen Vision System v3.0

![Apsen Farmacêutica](https://img.shields.io/badge/Apsen-Industrial_Vision-00b140?style=for-the-badge)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-FFD700?style=for-the-badge)
![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch_CUDA-EE4C2C?style=for-the-badge&logo=pytorch)

Sistema de Visão Computacional de alta performance desenvolvido para a **linha de embalagem industrial da Apsen Farmacêutica**. O sistema identifica medicamentos em tempo real cruzando inteligência artificial visual com leitura semântica de textos.

## 🚀 Arquitetura do Sistema

O pipeline de identificação trabalha de forma multithread para garantir máximo de FPS sem gargalos:

1. **YOLOv8 (Detecção Espacial)**: Redes neurais localizam as caixas em qualquer rotação ou posição na esteira e geram *Bounding Boxes*.
2. **Crop & Otimização**: A imagem (ROI) é recortada com precisão cirúrgica e reduzida para 250px para não sobrecarregar a memória da GPU.
3. **EasyOCR CUDA (Leitura de Dados)**: Um processamento pesado na GPU extrai todos os textos literais impressos na caixa. Suporta rotações de 0º, 90º, 180º e 270º.
4. **Fuzzy Matching**: Algoritmos de correlação de strings compensam ruídos na câmera (ex: "D1p1r0na") e cruzam a leitura com o banco de dados oficial (SKUs).

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