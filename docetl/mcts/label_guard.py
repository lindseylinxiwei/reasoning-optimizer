"""Keep ground-truth labels away from the optimizer agent and from candidate pipelines.

Some workloads store labels in the same records as the inputs (e.g. the Sustainability
splits carry company_name / economic_activity / document_category). If those records are
shown to the agent, or read by a pipeline, the search can copy the answers (search plan 13
grouped by the GT company_name and copied it into its output).

Evaluators read labels from the original split files, so the search only ever needs the
label-free view:
  - strip_labels() removes label fields from anything shown in a prompt,
  - sanitized_copy() writes a label-free copy of a dataset file for pipelines and samplers,
  - sanitize_pipeline() points a parsed pipeline's file datasets at those copies,
  - label_free_yaml() does the same for a pipeline YAML file.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

# Fields that hold ground truth, per workload. Inputs the task legitimately needs (ids,
# text, candidate label vocabularies such as BioDEX possible_labels) are not listed.
LABEL_FIELDS: Dict[str, Set[str]] = {
    "sustainability": {"company_name", "economic_activity", "document_category"},
    "biodex": {"ground_truth_reactions"},
    "medec": {
        "Error Flag", "Error Sentence", "Error Sentence ID", "Error Type",
        "Corrected Sentence", "Corrected Text",
    },
}

# Field names that look like labels in any workload; used only to warn.
_SUSPICIOUS = re.compile(r"ground[_ ]?truth|^gt[_ ]|_gt$|^label$|^labels$|^answer$", re.IGNORECASE)


def label_fields(dataset: Optional[str]) -> Set[str]:
    return LABEL_FIELDS.get((dataset or "").lower(), set())


def strip_labels(obj: Any, dataset: Optional[str]) -> Any:
    """Return obj with label fields removed from every dict it contains."""
    fields = label_fields(dataset)
    if not fields:
        return obj

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in fields}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(obj)


def warn_suspicious_fields(records: Any, dataset: Optional[str]) -> None:
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        return
    left = [k for k in records[0] if _SUSPICIOUS.search(k) and k not in label_fields(dataset)]
    if left:
        print(f"⚠️  Input fields {left} look like labels but are not in LABEL_FIELDS[{dataset!r}]; "
              f"they will be visible to the optimizer.")


def sanitized_copy(path: str, dataset: Optional[str], out_dir: Optional[str]) -> str:
    """Path to a copy of the JSON dataset at `path` without label fields.

    Returns `path` unchanged when the workload has no label fields, the file is missing or
    not a list of records, or no record has a label field. Copies are cached by source path,
    size and modification time.
    """
    fields = label_fields(dataset)
    if not fields or not path or not os.path.isfile(path):
        return path
    source = Path(path).resolve()
    if source.parent.name == "inputs_without_labels":
        return path
    stat = source.stat()
    digest = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
    target_dir = Path(out_dir or ".") / "inputs_without_labels"
    target = target_dir / f"{source.stem}_{digest}.json"
    if target.exists():
        return str(target)
    with open(source) as f:
        records = json.load(f)
    warn_suspicious_fields(records, dataset)
    if not isinstance(records, list) or not any(
        isinstance(r, dict) and fields & r.keys() for r in records
    ):
        return path
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(strip_labels(records, dataset), f)
    os.replace(tmp, target)
    return str(target)


def sanitize_pipeline(parsed_yaml: Dict, dataset: Optional[str], out_dir: Optional[str]) -> Dict:
    """Point every file dataset of a parsed pipeline at its label-free copy (in place)."""
    for ds in (parsed_yaml.get("datasets") or {}).values():
        if isinstance(ds, dict) and ds.get("type", "file") == "file" and ds.get("path"):
            ds["path"] = sanitized_copy(ds["path"], dataset, out_dir)
    return parsed_yaml


def label_free_yaml(yaml_path: str, dataset: Optional[str], out_dir: Optional[str]) -> str:
    """Path to a copy of the pipeline YAML whose datasets are label-free (or yaml_path if
    nothing needed changing). The copy is written to <out_dir>/inputs_without_labels/."""
    with open(yaml_path) as f:
        parsed = yaml.safe_load(f)
    before = json.dumps(parsed.get("datasets"), sort_keys=True)
    sanitize_pipeline(parsed, dataset, out_dir)
    if json.dumps(parsed.get("datasets"), sort_keys=True) == before:
        return yaml_path
    out = Path(out_dir or ".") / "inputs_without_labels" / Path(yaml_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(parsed, f, sort_keys=False)
    print(f"🔒 Using label-free inputs: {out}")
    return str(out)
