"""The search image holds no ground-truth file, and search-side evaluation goes to Modal."""
import builtins
import io
import json
from pathlib import Path

import pytest

from experiments.reasoning import label_isolation as li
# Imported here, not under the open() guard below: building the images locally reads the
# labeled train splits to write their label-free copies, which is not search-side code.
from experiments.reasoning import remote_eval, run_mcts, utils
from experiments.reasoning.evaluation.utils import get_evaluate_func, run_dataset_evaluation

DATA_DIR = li.EXPERIMENTS_DIR / "reasoning" / "data"
KNOWN_GT = [
    "reasoning/data/company_reports_gt.json",
    "reasoning/data/test/sustainability.json",
    "reasoning/data/val/sustainability.json",
    "reasoning/data/train_50/sustainability.json",
    "reasoning/data/CUAD-master_clauses.csv",
    "reasoning/data/sustainability_assurance_train_gt.json",
    "reasoning/data/train/biodex.json",
    "reasoning/data/train/medec.json",
]


@pytest.fixture(scope="module")
def labeled_files():
    return li.scan_labeled_files(li.EXPERIMENTS_DIR, skip_dirs=("__pycache__", ".venv", "node_modules", "biodex")) \
        + li.scan_labeled_files(li.DOCETL_DIR)


@pytest.fixture(scope="module")
def search_files():
    return (li.search_image_files(li.EXPERIMENTS_DIR, li.ignore_for_search_experiments)
            + li.search_image_files(li.DOCETL_DIR, li.ignore_for_search_docetl))


def test_scan_finds_the_known_ground_truth_files(labeled_files):
    found = {str(p.relative_to(li.EXPERIMENTS_DIR)) for p in labeled_files if li.EXPERIMENTS_DIR in p.parents}
    for rel in KNOWN_GT:
        if (li.EXPERIMENTS_DIR / rel).exists():
            assert rel in found, rel


def test_search_image_excludes_every_labeled_file(labeled_files, search_files):
    leaked = set(labeled_files) & set(search_files)
    assert not leaked, sorted(map(str, leaked))
    for rel in KNOWN_GT:
        assert li.ignore_for_search_experiments(Path(rel)), rel


def test_every_data_file_in_the_search_image_is_label_free(search_files):
    data = [p for p in search_files if p.suffix.lower() in li.DATA_SUFFIXES]
    assert data, "the search image should still hold the train inputs"
    for p in data:
        assert li.label_keys_in(p) == set(), p
        assert not li._GT_NAME.search(p.name), p


def test_search_image_from_experiments_is_code_pipelines_and_train_inputs(search_files):
    for p in search_files:
        if li.EXPERIMENTS_DIR not in p.parents:
            continue
        rel = p.relative_to(li.EXPERIMENTS_DIR)
        assert (p.suffix == ".py"
                or (p.suffix in (".yaml", ".yml") and any(x.startswith("pipelines") for x in rel.parts[:-1]))
                or rel.parts[:-1] == li.TRAIN_DIR), rel


def test_unreadable_data_files_count_as_labeled(tmp_path):
    bad = tmp_path / "inputs.json"
    bad.write_text("{not json")
    assert li.file_has_labels(bad)
    clean = tmp_path / "inputs2.json"
    clean.write_text(json.dumps([{"id": 1, "tot_text_raw": "x"}]))
    assert not li.file_has_labels(clean)
    gt = tmp_path / "inputs2_gt.json"
    gt.write_text(clean.read_text())
    assert li.file_has_labels(gt)


def test_label_free_train_copies_have_no_labels(tmp_path):
    pairs = li.label_free_train_copies(tmp_path)
    names = {Path(remote).name for _, remote in pairs}
    for name in ("biodex.json", "medec.json"):
        if (DATA_DIR / "train" / name).exists():
            assert name in names
    for local, remote in pairs:
        assert remote.startswith("/root/experiments/reasoning/data/train/")
        assert li.label_keys_in(local) == set()


class _Fake:
    def __init__(self, result):
        self.calls, self.result = [], result

    def remote(self, *args):
        self.calls.append(args)
        return self.result


@pytest.fixture
def label_free(monkeypatch):
    monkeypatch.setenv(li.LABEL_FREE_ENV, "1")
    labeled = {str(p.resolve()) for p in li.scan_labeled_files(DATA_DIR)}
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and str(Path(file).resolve()) in labeled:
            raise AssertionError(f"search-side code opened a ground-truth file: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)


def test_search_side_scoring_goes_to_modal(label_free, monkeypatch, tmp_path):
    fake = _Fake({"combined_score": 0.5})
    monkeypatch.setattr(remote_eval, "evaluate_remote", fake)
    out = tmp_path / "result.json"
    out.write_text("[]")
    metrics = get_evaluate_func("sustainability", mode="train")("docetl_preprint", str(out))
    assert metrics == {"combined_score": 0.5}
    assert fake.calls == [("sustainability", "train", "docetl_preprint", "result.json", b"[]")]


def test_search_side_final_evaluation_goes_to_modal(label_free, monkeypatch, tmp_path):
    fake = _Fake(([{"combined_score": 0.5}], {"evaluation_metrics.json": b"[1]", "plots/a.png": b"png"}))
    monkeypatch.setattr(remote_eval, "run_dataset_evaluation_remote", fake)
    res = tmp_path / "node_1" / "out.json"
    res.parent.mkdir()
    res.write_text("[]")

    class Node:
        result_path, cost, visits, value, mcts_accuracy, on_frontier = str(res), 1.5, 2, 0.1, 0.7, True

        def get_id(self):
            return "n1"

    eval_results = run_dataset_evaluation("sustainability", [Node()], tmp_path, method_name="docetl_mcts")
    assert eval_results == [{"combined_score": 0.5}]
    dataset, items, gt_path, method, root_cost = fake.calls[0]
    assert dataset == "sustainability" and gt_path is None and method == "docetl_mcts"
    assert items == [{"rel_path": "node_1/out.json", "content": b"[]", "is_node": True,
                      "info": {"node_id": "n1", "cost": 1.5, "visits": 2, "value": 0.1,
                               "mcts_accuracy": 0.7, "on_frontier": True}}]
    assert (tmp_path / "evaluation_metrics.json").read_bytes() == b"[1]"
    assert (tmp_path / "plots" / "a.png").read_bytes() == b"png"


def test_mcts_modal_function_uses_the_search_image():
    from modal._utils.async_utils import synchronizer

    image = run_mcts.run_mcts_remote.spec.image
    assert image is synchronizer._translate_in(utils.search_image)
    assert image is not synchronizer._translate_in(utils.image)
