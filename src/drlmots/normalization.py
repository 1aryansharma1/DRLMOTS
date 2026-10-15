"""Training-only bounded maximum scaling for synthetic state schema 1 (A-023)."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from drlmots.__main__ import configured_report
from drlmots.baselines import POLICIES
from drlmots.entities import Task
from drlmots.run_config import load_run_config
from drlmots.state import StateSchema


ALGORITHM = "training_max_clip_v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def _read_json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def _number(value):
    Task._validate_number("normalization value", value, allow_zero=True)
    return float(value)


def _layout(schema):
    binary, guards = set(), {}
    resources = len(schema.resource_units)
    for vm in range(len(schema.vm_ids)):
        start = 3 + resources + vm * (6 + 4 * resources)
        binary.update((start, start + 1))
        for resource in range(resources):
            offset = start + 6 + 4 * resource
            binary.update((offset, offset + 2))
            guards[offset + 1], guards[offset + 3] = offset, offset + 2
    binary.update(getattr(schema, "extra_binary_indices", ()))
    guards.update(getattr(schema, "extra_guards", {}))
    return frozenset(binary), guards


def _row(schema, values):
    values = tuple(_number(value) for value in values)
    if len(values) != schema.dimension:
        raise ValueError("state dimension does not match normalization schema")
    binary, guards = _layout(schema)
    if any(values[index] not in (0, 1) for index in binary):
        raise ValueError("binary state columns must be 0 or 1")
    if any(values[flag] == 0 and values[index] != 0 for index, flag in guards.items()):
        raise ValueError("unknown resource placeholders must be zero")
    return values


@dataclass(frozen=True)
class NormalizationContext:
    schema: StateSchema
    topology_sha256: str

    def __post_init__(self):
        if not isinstance(self.schema, StateSchema):
            raise TypeError("normalization context requires StateSchema")
        if (not isinstance(self.topology_sha256, str) or len(self.topology_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.topology_sha256)):
            raise ValueError("topology_sha256 must be a lowercase SHA-256")

    @classmethod
    def from_topology(cls, topology, resource_units=None):
        schema = StateSchema.from_topology(topology, resource_units)
        if any(vm.queue for vm in topology.vms):
            raise ValueError("normalization context requires empty initial VM queues")
        structure = asdict(topology)
        id_fields = {"hosts": "host_id", "vms": "vm_id", "fog_nodes": "fog_node_id",
                     "cloud_datacenters": "datacenter_id", "access_points": "access_point_id",
                     "end_nodes": "end_node_id"}
        for group, key in id_fields.items():
            for row in structure[group]:
                row.pop("queue", None)
                for name in row:
                    if name.endswith("_ids"):
                        row[name] = sorted(row[name])
            structure[group] = sorted(structure[group], key=lambda row: row[key])
        return cls(schema, _digest(structure))

    def manifest(self):
        return {"state_schema": self.schema.manifest(), "topology_sha256": self.topology_sha256}


@dataclass(frozen=True)
class TraceBatch:
    context: NormalizationContext
    states: tuple[tuple[float, ...], ...]
    split: str
    source_id: str
    provenance_json: str = "{}"

    def __post_init__(self):
        if not isinstance(self.context, NormalizationContext):
            raise TypeError("batch context must be NormalizationContext")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be nonempty")
        provenance = _read_json(self.provenance_json)
        if not isinstance(provenance, dict):
            raise ValueError("batch provenance must be a JSON object")
        object.__setattr__(self, "provenance_json", _canonical(provenance))
        object.__setattr__(self, "states", tuple(_row(self.context.schema, row) for row in self.states))


@dataclass(frozen=True)
class StateNormalizer:
    context: NormalizationContext
    maxima: tuple[float, ...]
    counts: tuple[int, ...]
    training_sources_json: str

    def __post_init__(self):
        if not isinstance(self.context, NormalizationContext):
            raise TypeError("normalizer context must be NormalizationContext")
        maxima = tuple(_number(value) for value in self.maxima)
        counts = tuple(self.counts)
        if len(maxima) != self.context.schema.dimension or len(counts) != len(maxima):
            raise ValueError("statistics dimension does not match schema")
        sources = _read_json(self.training_sources_json)
        if not isinstance(sources, list) or not sources:
            raise ValueError("training sources must be a nonempty list")
        ids = set()
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"source_id", "split", "vector_count", "provenance"}:
                raise ValueError("invalid training source fields")
            if (source["split"] != "train" or type(source["vector_count"]) is not int
                    or source["vector_count"] < 0 or not isinstance(source["provenance"], dict)
                    or not isinstance(source["source_id"], str) or not source["source_id"].strip()):
                raise ValueError("invalid training source provenance")
            if source["source_id"] in ids:
                raise ValueError("duplicate training source")
            ids.add(source["source_id"])
        sample_count = sum(source["vector_count"] for source in sources)
        if sample_count == 0:
            raise ValueError("fitting requires at least one training state")
        binary, guards = _layout(self.context.schema)
        for index, (maximum, count) in enumerate(zip(maxima, counts)):
            if type(count) is not int or not 0 <= count <= sample_count:
                raise ValueError("invalid feature sample count")
            if (index in binary and (maximum != 0 or count != 0)) or (count == 0 and maximum != 0):
                raise ValueError("statistics disagree with feature counts/layout")
            if index not in binary and index not in guards and count != sample_count:
                raise ValueError("unconditional numeric columns must count every training state")
        object.__setattr__(self, "maxima", maxima)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "training_sources_json", _canonical(sources))

    @property
    def scales(self):
        return tuple(value if value > 0 else 1.0 for value in self.maxima)

    def transform(self, context, state):
        if context != self.context:
            raise ValueError("incompatible normalization schema or topology")
        state = _row(context.schema, state)
        binary, guards = _layout(context.schema)
        result = []
        for index, (value, scale) in enumerate(zip(state, self.scales)):
            if index in binary:
                result.append(value)
            elif index in guards and state[guards[index]] == 0:
                result.append(0.0)
            else:
                result.append(1.0 if value >= scale else value / scale)
        return tuple(result)

    def transform_batch(self, batch):
        if not isinstance(batch, TraceBatch):
            raise TypeError("transform_batch requires TraceBatch")
        if batch.context != self.context:
            raise ValueError("incompatible normalization schema or topology")
        if batch.split != "train":
            task_hash = _read_json(batch.provenance_json).get("tasks_sha256")
            for source in _read_json(self.training_sources_json):
                if batch.source_id == source["source_id"] or (
                    task_hash is not None and task_hash == source["provenance"].get("tasks_sha256")
                ):
                    raise ValueError("held-out batch overlaps a declared training source/tasks")
        return tuple(self.transform(batch.context, state) for state in batch.states)

    def manifest(self):
        sources = _read_json(self.training_sources_json)
        payload = {
            "schema_version": 1, "kind": "synthetic_state_normalizer", "algorithm": ALGORITHM,
            "context": self.context.manifest(), "maxima": list(self.maxima), "counts": list(self.counts),
            "scales": list(self.scales), "training_vector_count": sum(s["vector_count"] for s in sources),
            "training_sources": sources,
        }
        return {**payload, "sha256": _digest(payload)}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, *, context):
        return cls.from_manifest(_read_json(Path(path).read_text(encoding="utf-8")), context=context)

    @classmethod
    def from_manifest(cls, payload, *, context):
        if not isinstance(context, NormalizationContext):
            raise TypeError("load requires a NormalizationContext")
        fields = {"schema_version", "kind", "algorithm", "context", "maxima", "counts", "scales",
                  "training_vector_count", "training_sources", "sha256"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid normalizer artifact fields")
        if (type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["kind"] != "synthetic_state_normalizer" or payload["algorithm"] != ALGORITHM):
            raise ValueError("unsupported normalization artifact version/algorithm")
        if _canonical(payload["context"]) != _canonical(context.manifest()):
            raise ValueError("incompatible normalization schema or topology")
        if payload["sha256"] != _digest({k: v for k, v in payload.items() if k != "sha256"}):
            raise ValueError("normalizer artifact checksum mismatch")
        result = cls(context, payload["maxima"], payload["counts"], _canonical(payload["training_sources"]))
        scales = tuple(_number(value) for value in payload["scales"])
        if (scales != result.scales or type(payload["training_vector_count"]) is not int
                or payload["training_vector_count"] != result.manifest()["training_vector_count"]):
            raise ValueError("normalizer scales/count disagree with statistics")
        return result


def fit_normalizer(training_batches):
    batches = tuple(training_batches)
    if not batches:
        raise ValueError("at least one training batch is required")
    if any(not isinstance(batch, TraceBatch) for batch in batches):
        raise TypeError("fitting requires TraceBatch values")
    if any(batch.split != "train" for batch in batches):
        raise ValueError("fitting accepts only train batches")
    context = batches[0].context
    if any(batch.context != context for batch in batches):
        raise ValueError("training batches have incompatible schema or topology")
    maxima, counts = [0.0] * context.schema.dimension, [0] * context.schema.dimension
    binary, guards = _layout(context.schema)
    for batch in batches:
        for state in batch.states:
            for index, value in enumerate(state):
                if index in binary or (index in guards and state[guards[index]] == 0):
                    continue
                maxima[index] = max(maxima[index], value)
                counts[index] += 1
    sources = [{"source_id": b.source_id, "split": "train", "vector_count": len(b.states),
                "provenance": _read_json(b.provenance_json)} for b in batches]
    return StateNormalizer(context, tuple(maxima), tuple(counts), _canonical(sources))


def capture_batch(config, *, split, policy="least_queue_fcfs"):
    report = configured_report(config, policy=policy, capture_decisions=True)
    context = NormalizationContext.from_topology(config.topology, config.resource_units)
    provenance = {
        "run_configuration": report["run_configuration"], "workload_sha256": report["workload"]["sha256"],
        "tasks_sha256": _digest(report["workload"]["tasks"]), "generator": report["workload"].get("generator"),
        "software_version": report["version"], "policy": policy,
    }
    source_id = _digest({key: provenance[key] for key in ("run_configuration", "workload_sha256", "policy")})
    batch = TraceBatch(context, tuple(d["state"] for d in report["decision_trace"]["decisions"]),
                       split, source_id, _canonical(provenance))
    return batch, report


def main():
    parser = argparse.ArgumentParser(description="Fit training-only state scaling or apply a saved normalizer")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit only explicitly supplied training configurations")
    fit.add_argument("--train-config", type=Path, action="append", required=True)
    apply = commands.add_parser("apply", help="Transform a held-out run without fitting")
    apply.add_argument("--config", type=Path, required=True)
    apply.add_argument("--normalizer", type=Path, required=True)
    apply.add_argument("--split", choices=("validation", "test"), default="test")
    for command in (fit, apply):
        command.add_argument("--policy", choices=POLICIES, default="least_queue_fcfs")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = args.train_config if args.command == "fit" else [args.config, args.normalizer]
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("output must not overwrite an input configuration or normalizer")
        if args.command == "fit":
            batches = [capture_batch(load_run_config(path), split="train", policy=args.policy)[0]
                       for path in args.train_config]
            normalizer = fit_normalizer(batches)
            normalizer.save(args.output)
            print(f"Saved normalizer: {args.output.resolve()} ({normalizer.manifest()['training_vector_count']} training states)")
        else:
            batch, report = capture_batch(load_run_config(args.config), split=args.split, policy=args.policy)
            normalizer = StateNormalizer.load(args.normalizer, context=batch.context)
            transformed = normalizer.transform_batch(batch)
            for decision, vector in zip(report["decision_trace"]["decisions"], transformed):
                decision["normalized_state"] = vector
            report["normalization"] = {"algorithm": ALGORITHM, "artifact_sha256": normalizer.manifest()["sha256"],
                                       "split": batch.split, "source_id": batch.source_id}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Saved normalized {batch.split} report: {args.output.resolve()} ({len(transformed)} states)")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
