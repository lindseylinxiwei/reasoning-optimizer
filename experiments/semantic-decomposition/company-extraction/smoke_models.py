"""Local extraction smoke test only; no experiment dataset or paid APIs."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

TEXT = 'Apple published its annual sustainability report. The report describes Apple\'s emissions reductions and renewable energy initiatives. Microsoft is mentioned as an industry peer.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gliner', 'gliner25'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    model_id = 'urchade/gliner_small-v2.1' if args.model == 'gliner' else 'fastino/gliner2.5-base-v1'
    report = {'model': model_id, 'text': TEXT, 'labels': ['organization'], 'cpu_threads': 4,
              'torch': torch.__version__, 'mps_available': torch.backends.mps.is_available(),
              'purpose': 'Synthetic smoke test; not accuracy or full-document latency evaluation', 'devices': []}
    for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
        entry = {'device': device}
        report['devices'].append(entry)
        try:
            start = time.perf_counter()
            if args.model == 'gliner':
                from gliner import GLiNER
                model = GLiNER.from_pretrained(model_id).to(device)
                def extract():
                    return model.predict_entities(TEXT, ['organization'], threshold=0.5)
            else:
                from gliner2 import AutoExtractor
                model = AutoExtractor.from_pretrained(model_id, map_location=device)
                def extract():
                    return model.extract_entities(TEXT, ['organization'], include_spans=True, include_confidence=True)['entities']['organization']
            model.eval()
            entry['load_seconds_including_download_if_needed'] = time.perf_counter() - start
            entry['parameter_count'] = sum(p.numel() for p in model.parameters())
            entry['actual_device'] = str(next(model.parameters()).device)
            durations = []
            for iteration in range(3):
                if device == 'mps': torch.mps.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    entities = extract()
                if device == 'mps': torch.mps.synchronize()
                durations.append(time.perf_counter()-start)
                assert entities, 'No organization candidates returned'
                assert all(TEXT[x['start']:x['end']] == x['text'] for x in entities), 'Invalid source offsets'
                assert any(x['text'] == 'Apple' for x in entities), 'Expected subject-company candidate missing'
            entry.update(status='success', inference_seconds=durations, entities=entities, span_copy_valid=True)
            del model
            gc.collect()
            if device == 'mps': torch.mps.empty_cache()
        except Exception as exc:
            entry.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(entry), flush=True)
    if not any(x['status']=='success' for x in report['devices']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
