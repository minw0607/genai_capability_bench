"""Dataset registry and loaders."""

from genai_capability_bench.datasets.registry import (
    DatasetSpec,
    get_dataset_spec,
    list_dataset_specs,
    materialize_dataset,
)

__all__ = [
    "DatasetSpec",
    "get_dataset_spec",
    "list_dataset_specs",
    "materialize_dataset",
]

