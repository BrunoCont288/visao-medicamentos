# Apsen Vision System

Sistema de Visão Computacional para identificação industrial de medicamentos em tempo real. Desenvolvido para a linha de embalagem da Apsen Farmacêutica.

## Funcionalidades
- **Detecção via YOLOv8**: Modelo treinado em dataset sintético transparente para detectar qualquer orientação e rotação.
- **OCR com EasyOCR**: Extração de texto focada usando Inteligência Artificial paralela via GPU/CUDA.
- **Dashboard Dinâmico**: Interface visual com FPS, Confiança e status do medicamento.
- **Fuzzy Matching**: Sistema perdoa erros de leitura (ex: "D1p1rona") e cruza com a base de dados JSON.

## Stack
- Python 3.12
- Ultralytics YOLOv8
- OpenCV (cv2)
- EasyOCR (PyTorch CUDA)
- thefuzz

## Instalação

\`\`\`bash
# 1. Instale o PyTorch versão CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Instale as bibliotecas
pip install opencv-python ultralytics easyocr thefuzz
\`\`\`

## Execução

\`\`\`bash
python main.py
\`\`\`