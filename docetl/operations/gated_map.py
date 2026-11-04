"""gated_map: a map that runs only on documents whose gate field is true.

For gating an expensive generation behind a cheap decision (for example a Jev presence
question that writes `gate_key`). Documents whose `gate_key` is false keep their
fields and get None for every output field, so no document is dropped and no LLM call is
made for it. Everything else behaves exactly like `map`.
"""
from docetl.operations.map import MapOperation


class GatedMapOperation(MapOperation):
    class schema(MapOperation.schema):
        type: str = "gated_map"
        gate_key: str

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        key = self.config["gate_key"]
        fields = list((self.config.get("output") or {}).get("schema") or {})
        chosen = [i for i, d in enumerate(input_data) if d.get(key)]
        results, cost = super().execute([input_data[i] for i in chosen]) if chosen else ([], 0.0)
        by_index = dict(zip(chosen, results))
        out = []
        for i, doc in enumerate(input_data):
            out.append(by_index[i] if i in by_index else {**doc, **{f: None for f in fields}})
        return out, cost
