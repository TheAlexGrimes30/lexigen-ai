from dataclasses import dataclass


@dataclass(frozen=True)
class HybridRetrieverConfig:
    alpha: float = 0.8
    graph_weight: float = 0.15
    pool_multiplier: int = 8
    max_pool_size: int = 80
    min_text_len: int = 40