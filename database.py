"""
Banco de dados de medicamentos com fuzzy matching.

Carrega os medicamentos do JSON e fornece busca por similaridade
para identificar nomes a partir de texto OCR.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from thefuzz import fuzz, process

from config import CONFIG, get_logger


logger = get_logger("Database")



class MedicineDatabase:

    def __init__(self, json_path: str) -> None:
        self._path = Path(json_path)
        self._data: List[Dict] = []
        self._all_names: List[str] = []
        self._name_to_official: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
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
