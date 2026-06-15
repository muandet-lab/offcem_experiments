"""YAML configuration and experimental-design planning."""
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Dict
from typing import List
import uuid

import numpy as np
from scipy.stats import qmc
import yaml


def load_config(path: str) -> Dict:
    with open(path) as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a YAML mapping")
    return config


def build_cells(config: Dict) -> List[Dict]:
    defaults = config.get("defaults", {})
    design = config.get("design", {"type": "explicit", "points": [{}]})
    design_type = design.get("type", "explicit")
    if design_type == "explicit":
        points = design.get("points", [{}])
    elif design_type == "sobol":
        points = _sobol_points(design)
    else:
        raise ValueError(f"Unknown design.type={design_type!r}")

    cells = []
    for point in points:
        merged = copy.deepcopy(defaults)
        _deep_update(merged, point)
        cells.append(merged)
    return cells


def config_hash(cell: Dict, length: int = 16) -> str:
    payload = json.dumps(cell, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def task_count(cells: List[Dict], n_seeds: int) -> int:
    return len(cells) * n_seeds


def task_coordinates(
    task_index: int,
    n_cells: int,
    n_seeds: int,
):
    total = n_cells * n_seeds
    if task_index < 0 or task_index >= total:
        raise ValueError(
            f"task_index must be in [0, {total}), got {task_index}"
        )
    return task_index // n_seeds, task_index % n_seeds


def task_index(cell_index: int, seed_index: int, n_seeds: int) -> int:
    if cell_index < 0:
        raise ValueError("cell_index must be non-negative")
    if seed_index < 0 or seed_index >= n_seeds:
        raise ValueError(
            f"seed_index must be in [0, {n_seeds}), got {seed_index}"
        )
    return cell_index * n_seeds + seed_index


def checkpoint_name(cell: Dict, seed_index: int) -> str:
    return f"{config_hash(cell)}_seed{seed_index:04d}.json"


def write_manifest(
    config_path: str,
    config: Dict,
    cells: List[Dict],
    output: Path,
    n_seeds: int,
):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_path": str(Path(config_path).resolve()),
        "config_hash": config_hash(config),
        "n_cells": len(cells),
        "n_seeds": n_seeds,
        "n_tasks": task_count(cells, n_seeds),
        "cells": [
            {"cell_id": config_hash(cell), "config": cell}
            for cell in cells
        ],
        "tasks": [
            {
                "task_index": task_index(cell_index, seed_index, n_seeds),
                "cell_index": cell_index,
                "seed_index": seed_index,
                "cell_id": config_hash(cell),
                "checkpoint": checkpoint_name(cell, seed_index),
            }
            for cell_index, cell in enumerate(cells)
            for seed_index in range(n_seeds)
        ],
    }
    _atomic_json_dump(output / "manifest.json", manifest)

    lines = ["task_index\tcell_index\tseed_index\tcell_id\tcheckpoint"]
    lines.extend(
        "\t".join(
            str(task[field])
            for field in (
                "task_index",
                "cell_index",
                "seed_index",
                "cell_id",
                "checkpoint",
            )
        )
        for task in manifest["tasks"]
    )
    _atomic_text_write(output / "tasks.tsv", "\n".join(lines) + "\n")


def estimate_runtime_seconds(config: Dict, cells: List[Dict]) -> float:
    runtime = config.get("runtime", {})
    reference_seconds = float(runtime.get("reference_seconds", 0.0))
    if reference_seconds <= 0:
        return 0.0
    reference_n = float(runtime.get("reference_n_rounds", 3000))
    reference_actions = float(runtime.get("reference_n_actions", 1000))
    seeds = int(config.get("execution", {}).get("n_seeds", 1))
    total = 0.0
    for cell in cells:
        scale = (
            float(cell["n_rounds"])
            / reference_n
            * float(cell["dataset"]["n_actions"])
            / reference_actions
        )
        partitions = len(cell.get("partitions", []))
        total += reference_seconds * scale * max(partitions, 1) * seeds
    return total


def _sobol_points(design: Dict) -> List[Dict]:
    n_cells = int(design["n_cells"])
    factors = design.get("factors", {})
    names = list(factors)
    extra_dims = 4 if design.get("sample_reward_simplex", False) else 0
    sampler = qmc.Sobol(
        d=len(names) + extra_dims,
        scramble=True,
        seed=int(design.get("seed", 0)),
    )
    power = int(np.ceil(np.log2(max(n_cells, 1))))
    samples = sampler.random_base2(power)[:n_cells]
    points = []
    for sample in samples:
        point = {}
        for index, name in enumerate(names):
            spec = factors[name]
            value = _map_factor(sample[index], spec)
            _set_dotted(point, name, value)
        if extra_dims:
            raw = -np.log(np.maximum(sample[-4:], 1e-12))
            alpha = (raw / raw.sum()).tolist()
            _set_dotted(point, "reward.alpha", alpha)
        points.append(point)
    return points


def _map_factor(value: float, spec):
    if isinstance(spec, list):
        return spec[min(int(value * len(spec)), len(spec) - 1)]
    if "values" in spec:
        values = spec["values"]
        return values[min(int(value * len(values)), len(values) - 1)]
    low, high = float(spec["min"]), float(spec["max"])
    mapped = low + value * (high - low)
    if spec.get("scale") == "log":
        mapped = np.exp(np.log(low) + value * (np.log(high) - np.log(low)))
    if spec.get("type") == "int":
        return int(round(mapped))
    return float(mapped)


def _set_dotted(target: Dict, dotted: str, value):
    current = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _deep_update(target: Dict, source: Dict):
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _atomic_json_dump(path: Path, value: Dict):
    _atomic_text_write(path, json.dumps(value, indent=2) + "\n")


def _atomic_text_write(path: Path, value: str):
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "w") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
