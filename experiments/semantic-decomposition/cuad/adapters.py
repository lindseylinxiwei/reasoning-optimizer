"""Small adapters around the existing pinned GLiNER loader and Jev client."""
import importlib.util
import json
from pathlib import Path

from pipeline import digest, save, windows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_client():
    return load_module('_cuad_shared', HERE.parent / 'company-extraction/pipeline.py')


class Extractor:
    def __init__(self, schema, device='cpu', threshold=.2, window_tokens=1000,
                 overlap_tokens=250, batch_size=2):
        self.loader = shared_client().LocalExtractor('gliner25', device, threshold)
        self.model = self.loader.model
        self.schema, self.threshold = schema, threshold
        self.size, self.overlap, self.batch_size = window_tokens, overlap_tokens, batch_size
        self.schema_dict = self.model.create_schema().entities(schema).build()
        self.max_encoded_tokens = self.model.config.max_len

    def fits(self, text):
        # Encode text AND the real schema without truncation, using the same processor as inference.
        batch = self.model.processor.collate_fn_inference(
            [(text, self.schema_dict)], max_len=None, error_policy='raise', architecture='boundary')
        return batch.input_ids.shape[1] <= self.max_encoded_tokens

    def __call__(self, text):
        offsets = self.model.processor.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)['offset_mapping']
        regions = windows(text, offsets, self.size, self.overlap, self.fits)
        chunks = [text[r['start']:r['end']] for r in regions]
        predictions = self.model.batch_extract_entities(
            chunks, self.schema, batch_size=self.batch_size, threshold=self.threshold,
            include_spans=True, include_confidence=True, overlap_policy='allow')
        if self.loader.device == 'mps':
            self.loader.torch.mps.synchronize()
        return regions, predictions


class Verifier:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.client = shared_client().JevSelector(self.folder / 'calls.jsonl')

    def fits(self, state, questions):
        try:
            self.client._pack(state, questions)
            return True
        except ValueError:
            return False

    def __call__(self, state, questions):
        answers = {}
        for group in self.client._pack(state, questions):
            identity = {'model': 'jev-1.13.0', 'state': state, 'questions': group}
            key = digest(json.dumps(identity, sort_keys=True, ensure_ascii=False))
            checkpoint = self.folder / (key + '.json')
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text())
            else:
                result = self.client._request(state, state, group, 1, 1)
                save(checkpoint, result)
            if set(result['answers']) != set(group):
                raise ValueError('Jev response does not cover the exact question IDs')
            answers.update(result['answers'])
        return answers


def api_metrics(folder):
    rows = [json.loads(line) for p in Path(folder).glob('*/requests/calls.jsonl')
            for line in p.read_text().splitlines()]
    return {'api_calls': len(rows),
            'known_api_cost_usd': sum(r['cost_usd'] for r in rows if r.get('cost_usd') is not None),
            'unknown_cost_calls': sum(r.get('cost_usd') is None for r in rows),
            'input_tokens': sum(r.get('response', {}).get('usage', {}).get('input_tokens', 0) for r in rows),
            'api_wall_seconds': sum(r['wall_seconds'] for r in rows)}
