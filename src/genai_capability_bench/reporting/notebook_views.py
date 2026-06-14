"""Notebook-friendly display table builders."""

from __future__ import annotations

import pandas as pd

from genai_capability_bench.core.schemas import Capability
from genai_capability_bench.datasets.registry import list_dataset_specs


def dataset_catalog_table(capability: Capability | None = None) -> pd.DataFrame:
    """Return a concise dataset catalog for notebooks."""

    rows = []
    for spec in list_dataset_specs(capability):
        rows.append(
            {
                "Dataset Key": spec.key,
                "Name": spec.display_name,
                "Source": spec.source_type,
                "Default Split": spec.default_split,
                "Primary Capability": spec.capability.value,
                "Context Needed": "Yes" if spec.requires_context else "No",
                "Description": spec.description,
                "Notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def model_config_table(models) -> pd.DataFrame:
    """Return model configuration display table."""

    return pd.DataFrame(
        [
            {
                "Name": m.name,
                "Provider": m.provider,
                "Model / Deployment": m.model,
                "API Version": m.api_version or "",
                "Temperature": m.temperature,
                "Max Tokens": m.max_tokens,
            }
            for m in models
        ]
    )

