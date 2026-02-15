from __future__ import annotations

import hashlib
import json


def compute_merkle_id(
    code_manifest: dict[str, str],
    params: dict[str, object],
    input_merkle_ids: dict[str, str],
) -> str:
    data = {
        "code": code_manifest,
        "params": params,
        "inputs": sorted(input_merkle_ids.items()),
    }
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
