"""Run off-the-shelf CUAD extraction, optionally followed by Jev verification."""
import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from adapters import Extractor, HERE, ROOT, Verifier, api_metrics
from pipeline import collect_candidates, digest, load_schema, save, serialize, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reasoning/data/train/cuad.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--threshold', type=float, default=.2)
    parser.add_argument('--window-tokens', type=int, default=1000)
    parser.add_argument('--overlap-tokens', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    if (args.limit < 1 or args.batch_size < 1 or not 0 <= args.threshold <= 1
            or not 0 <= args.overlap_tokens < args.window_tokens):
        parser.error('Invalid limit, batch size, threshold, or window settings')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    schema = load_schema(ROOT / 'experiments/reasoning/pipelines/cuad.yaml')
    data = json.loads(args.input.read_text())[:args.limit]
    if not data or len({str(r['id']) for r in data}) != len(data):
        raise ValueError('Expected unique document IDs')
    # Only these source fields enter inference. Ground truth is loaded by evaluate.py, separately.
    records = [{k: row[k] for k in ('id', 'name', 'document')} for row in data]
    if any(not isinstance(r['document'], str) or not r['document'].strip() for r in records):
        raise ValueError('Contracts must be nonempty strings')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'extract_only'}
    sources = [HERE / name for name in ('pipeline.py', 'adapters.py', 'run.py')]
    sources += [HERE.parent / 'company-extraction' / name
                for name in ('pipeline.py', 'downloaded_revisions.json')]
    protocol.update(schema=schema, input_sha256=digest(json.dumps(records, sort_keys=True)),
                    code_sha256={str(p.relative_to(ROOT)): digest(p.read_text()) for p in sources})
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Existing run has a different protocol; use a new output directory')
    save(path, protocol)
    save(args.output / 'input.json', records)
    extractor = None
    baseline, verified, timings = [], [], []
    started = time.perf_counter()
    try:
        for row in records:
            folder = args.output / digest(str(row['id']))[:16]
            folder.mkdir(exist_ok=True)
            text, elapsed = row['document'], {'id': row['id']}
            path = folder / 'extraction.json'
            if path.exists():
                extraction = json.loads(path.read_text())
            else:
                if extractor is None:
                    loaded = time.perf_counter()
                    extractor = Extractor(schema, args.device, args.threshold, args.window_tokens,
                                          args.overlap_tokens, args.batch_size)
                    model_info = {'model': extractor.loader.model_id, 'revision': extractor.loader.revision,
                                  'load_seconds': time.perf_counter() - loaded,
                                  'max_encoded_tokens': extractor.max_encoded_tokens,
                                  'overlap_policy': 'allow',
                                  'versions': {name: version(name) for name in ('gliner2', 'torch', 'transformers')}}
                    save(args.output / 'model.json', model_info)
                t = time.perf_counter()
                regions, predictions = extractor(text)
                candidates, invalid = collect_candidates(text, regions, predictions, schema)
                extraction = {'regions': regions, 'candidates': candidates, 'invalid_spans': invalid,
                              'raw_predictions': predictions, 'seconds': time.perf_counter() - t}
                save(path, extraction)
            candidates = extraction['candidates']
            elapsed['extraction_seconds'] = extraction['seconds']
            item = {k: row[k] for k in ('id', 'name')}
            baseline.append({**item, 'clauses': serialize(text, candidates, schema)})
            save(args.output / 'gliner.json', baseline)
            if not args.extract_only:
                t = time.perf_counter()
                decisions, trace = verify(text, candidates, extraction['regions'], schema,
                                          Verifier(folder / 'requests'))
                elapsed['verification_seconds_this_run'] = time.perf_counter() - t
                save(folder / 'verification.json', {'decisions': decisions, 'trace': trace})
                verified.append({**item, 'clauses': serialize(text, candidates, schema, decisions)})
                save(args.output / 'gliner_jev.json', verified)
            timings.append(elapsed)
            print(f"{row['id']}: {len(extraction['regions'])} windows, {len(candidates)} candidates, "
                  f"{len(extraction['invalid_spans'])} invalid spans", flush=True)
    finally:
        save(args.output / 'metrics.json', {**api_metrics(args.output), 'documents_completed': len(timings),
             'elapsed_seconds_this_run': time.perf_counter() - started, 'timings': timings,
             'local_compute_cost': 'Not monetized; measured separately in seconds.',
             'api_price_assumption': 'Existing client: USD 0.042 per million returned input tokens.'})


if __name__ == '__main__':
    main()
