from enum import Enum


class TechClass(Enum):
    cpc = "cpc"
    ipc = "ipc"


class PrimaryKey(Enum):
    publication_number = "publication_number"
    family_id = "family_id"


class CitationKind(Enum):
    backward = "back"
    forward = "for"


class CitationExpansionLevel(Enum):
    L1 = "L1"
    L2 = "L2"


class OverlapAnalysisKind(Enum):
    pairwise = "pairwise"
    batch = "batch"


class OverlapAnalysisAxis(Enum):
    technologies = "technologies"
    configs = "configs"
