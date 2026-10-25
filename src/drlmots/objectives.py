"""Explicit synthetic energy/risk model shared by live rewards and evaluation.

This is a documented interpretation of the paper, not measured hardware energy
or a fault-injection/redundancy implementation. All times are milliseconds.
"""

from dataclasses import dataclass
import math

from drlmots.entities import Task


@dataclass(frozen=True)
class VMObjectiveProfile:
    vm_id: str
    active_power_w: float
    transmission_j_per_kb: float
    failure_rate_per_s: float

    def __post_init__(self):
        if not isinstance(self.vm_id, str) or not self.vm_id.strip():
            raise ValueError("objective VM ID must be nonempty")
        for name in ("active_power_w", "transmission_j_per_kb", "failure_rate_per_s"):
            Task._validate_number(name, getattr(self, name), allow_zero=True)

    def manifest(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class MultiObjectiveConfig:
    latency_weight: float
    energy_weight: float
    failure_risk_weight: float
    energy_scale_j: float
    vms: tuple[VMObjectiveProfile, ...]

    def __post_init__(self):
        weights = (self.latency_weight, self.energy_weight, self.failure_risk_weight)
        for value in weights:
            Task._validate_number("objective weight", value, allow_zero=True)
        if not math.isclose(math.fsum(weights), 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("objective weights must sum to one")
        Task._validate_number("energy_scale_j", self.energy_scale_j, allow_zero=False)
        profiles = tuple(self.vms)
        if not profiles or any(not isinstance(p, VMObjectiveProfile) for p in profiles):
            raise ValueError("objective profiles must contain VMObjectiveProfile values")
        if len({p.vm_id for p in profiles}) != len(profiles):
            raise ValueError("duplicate objective VM profile")
        object.__setattr__(self, "vms", tuple(sorted(profiles, key=lambda p: p.vm_id)))

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {"schema_version", "weights", "energy_scale_j", "vms"}:
            raise ValueError("invalid multi-objective fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported objective schema")
        weights = data["weights"]
        if not isinstance(weights, dict) or set(weights) != {"latency", "energy", "failure_risk"}:
            raise ValueError("objective weights require latency, energy and failure_risk")
        if not isinstance(data["vms"], (list, tuple)):
            raise ValueError("objective vms must be a list")
        profiles = []
        for item in data["vms"]:
            if not isinstance(item, dict) or set(item) != set(VMObjectiveProfile.__dataclass_fields__):
                raise ValueError("invalid objective VM profile fields")
            profiles.append(VMObjectiveProfile(**item))
        return cls(weights["latency"], weights["energy"], weights["failure_risk"],
                   data["energy_scale_j"], tuple(profiles))

    def validate_vm_ids(self, vm_ids):
        if {p.vm_id for p in self.vms} != set(vm_ids):
            raise ValueError("objective profiles must exactly match the environment VM IDs")

    def manifest(self):
        return {"schema_version": 1,
                "weights": {"latency": self.latency_weight, "energy": self.energy_weight,
                            "failure_risk": self.failure_risk_weight},
                "energy_scale_j": self.energy_scale_j, "vms": [p.manifest() for p in self.vms]}

    def profile(self, vm_id):
        for item in self.vms:
            if item.vm_id == vm_id:
                return item
        raise ValueError("missing objective VM profile: " + str(vm_id))


def normalized_cost(value, scale):
    """Bounded x/(scale+x), avoiding overflow in the denominator."""
    Task._validate_number("objective value", value, allow_zero=True)
    Task._validate_number("objective scale", scale, allow_zero=False)
    return 1 / (1 + scale / value) if value >= scale else (value / scale) / (1 + value / scale)


def task_objectives(profile, *, service_ms, data_size_kb, latency_ms, network):
    for name, value in (("service_ms", service_ms), ("data_size_kb", data_size_kb), ("latency_ms", latency_ms)):
        Task._validate_number(name, value, allow_zero=True)
    if type(network) is not bool or latency_ms < service_ms - 1e-9:
        raise ValueError("invalid network flag or latency shorter than service")
    compute = profile.active_power_w * (service_ms / 1000)
    transmission = profile.transmission_j_per_kb * data_size_kb if network else 0.0
    energy = math.fsum((compute, transmission))
    Task._validate_number("task energy joules", energy, allow_zero=True)
    # Constant destination hazard over arrival-to-finish exposure. Analytical only:
    # no failure changes queue execution, and no backup or recovery is simulated.
    risk = -math.expm1(-profile.failure_rate_per_s * (latency_ms / 1000))
    return {"latency_ms": latency_ms, "service_ms": service_ms,
            "compute_energy_j": compute, "transmission_energy_j": transmission,
            "total_energy_j": energy, "failure_risk": risk, "success_probability": 1 - risk}


def objective_score(values, config, latency_scale_ms):
    costs = {"latency": normalized_cost(values["latency_ms"], latency_scale_ms),
             "energy": normalized_cost(values["total_energy_j"], config.energy_scale_j),
             "failure_risk": values["failure_risk"]}
    cost = math.fsum((config.latency_weight * costs["latency"],
                      config.energy_weight * costs["energy"],
                      config.failure_risk_weight * costs["failure_risk"]))
    return {"normalized_costs": costs, "weighted_cost": cost, "utility": 1 - cost}


def frame_objectives(frame, action, config, latency_scale_ms):
    vm = frame.vms[action]
    estimate = vm.estimate
    if estimate is None:
        raise ValueError("cannot evaluate an infeasible VM")
    # Zero-network frames have exactly zero upload time; schema-3 uploads are
    # strictly positive because tasks/data/bandwidth are positive.
    values = task_objectives(config.profile(vm.vm_id), service_ms=estimate.service_ms,
                             data_size_kb=frame.data_size_kb, latency_ms=estimate.completion_duration_ms,
                             network=vm.upload_ms > 0)
    return {**values, **objective_score(values, config, latency_scale_ms)}


def evaluate_objectives(report, config, *, latency_scale_ms=10.0, rejection_penalty=1.0):
    """Recompute metrics from realized simulator outcomes, not predicted rewards."""
    config.validate_vm_ids(vm["vm_id"] for vm in report["run_configuration"]["topology"]["vms"])
    tasks = {item["task_id"]: item for item in report["workload"]["tasks"]}
    if len(tasks) != len(report["outcomes"]) or len({r["task_id"] for r in report["outcomes"]}) != len(tasks):
        raise ValueError("objective report must account for each input task once")
    rows = []
    network = report["run_configuration"].get("schema_version") == 3
    for outcome in report["outcomes"]:
        task = tasks[outcome["task_id"]]
        if outcome["status"] != "completed":
            if outcome["vm_id"] is not None or outcome["finish_time_ms"] is not None:
                raise ValueError("unsupported failed execution; objective model expects admission rejection")
            rows.append({"task_id": outcome["task_id"], "vm_id": None, "admitted": False,
                         "utility": -float(rejection_penalty), "success_probability": 0.0})
            continue
        values = task_objectives(config.profile(outcome["vm_id"]),
                                 service_ms=outcome["finish_time_ms"] - outcome["start_time_ms"],
                                 data_size_kb=task["data_size_kb"],
                                 latency_ms=outcome["finish_time_ms"] - outcome["arrival_time_ms"], network=network)
        rows.append({"task_id": outcome["task_id"], "vm_id": outcome["vm_id"], "admitted": True,
                     **values, **objective_score(values, config, latency_scale_ms)})
    admitted = [row for row in rows if row["admitted"]]
    total = len(rows)
    compute = math.fsum(row["compute_energy_j"] for row in admitted)
    transmission = math.fsum(row["transmission_energy_j"] for row in admitted)
    energy = math.fsum((compute, transmission))
    expected = math.fsum(row["success_probability"] for row in admitted)
    return {"model": config.manifest(),
            "scope": "synthetic active compute + input transmission; analytical destination failure risk; no injected faults, backups, idle power or network contention",
            "metrics": {"submitted_count": total, "completed_count": len(admitted),
                        "rejected_count": total - len(admitted),
                        "completion_rate": len(admitted) / total if total else None,
                        "makespan_ms": report["makespan_ms"],
                        "successful_mean_latency_ms": report["average_completion_time_ms"],
                        "compute_energy_j": compute, "transmission_energy_j": transmission, "total_energy_j": energy,
                        "energy_per_completed_task_j": energy / len(admitted) if admitted else None,
                        "expected_successful_tasks": expected,
                        "expected_success_rate_all_arrivals": expected / total if total else None,
                        "expected_execution_failures": math.fsum(row["failure_risk"] for row in admitted),
                        "mean_failure_risk_admitted": math.fsum(row["failure_risk"] for row in admitted) / len(admitted) if admitted else None,
                        "energy_per_expected_success_j": energy / expected if expected else None,
                        "mean_all_arrival_utility": math.fsum(row["utility"] for row in rows) / total if total else None},
            "tasks": rows}
