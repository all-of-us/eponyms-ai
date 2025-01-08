# coding=utf-8

# Paths
from pathlib import Path

curr_path = Path(__file__)
resources_path = curr_path.parent
base_path = resources_path.parent
core_path = base_path / 'core'
data_path = base_path / 'data'
guideline_path = resources_path / 'guideline.md'

ANNOTATED_COLS = [
    "concept_name", "tokenized_concept_name", "concept_code", "concept_id",
    "eponym", "descriptor", "He", "Hd", "Hc", "Ce", "Cd", "Cc", "Le", "Ld", "Lc",
    "Te", "Td", "Fe", "Fd", "Fc", "Ep", "De", "1e", "0e", "1d", "0d", "Che", "Chd",
    "Pe", "Pd", "N1", "N2", "N3", "N4"
]

DEFAULT_COLS = [
    "concept_name", "tokenized_concept_name", "concept_code", "concept_id",
    "eponym", "suffix", "r"
]

