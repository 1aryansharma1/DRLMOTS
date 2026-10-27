"""Observation-only multi-objective teacher and masked behavior-cloning warm start.

This is BC followed by basic DQN, not a reproduction of DQfD. The original
native multi-objective simulator policy is preserved as a separate comparator.
"""
import math

import torch
from torch.nn import functional as F

from drlmots.enhanced_dqn import _legal_actions
from drlmots.enhanced_rl_contract import EncodedObservation
from drlmots.enhanced_sanity import network_digest
from drlmots.objective_model import integer, number
from drlmots.predictive_dqn import PREDICTIONS, PredictiveDQN


TEACHER = dict(schema_version=1, kind="observed_multi_objective_teacher",
    inputs="contract-4 immutable observation and public configuration only",
    estimates="unchanged predictive feature schema 1; delayed sample-and-hold estimates",
    score="w_time*predicted_latency/decision_time_scale + w_energy*predicted_incremental_energy/decision_energy_scale + w_failure*conditional_failure_probability",
    choice="legal actions only; if any predicts deadline met, filter to those; minimum score, fog on exact tie, VM ID",
    restrictions="no simulator state, future arrivals, outage trace, actual outcome or native estimates",
    distinction="same multi-objective scoring rule with observed estimates; not the native event-loop oracle")


def teacher_choice(agent, observation):
    legal = _legal_actions(agent.contract, observation)
    if not legal:
        return None, ()
    with torch.inference_mode():
        predictions = agent.online.predictions(torch.tensor(observation.values, dtype=torch.float32))
    predictions = predictions.reshape(len(agent.contract.vm_ids), len(PREDICTIONS)).tolist()
    columns = {p[0]: i for i, p in enumerate(PREDICTIONS)}
    o = agent.contract.context["objective"]
    w, s = o["weights"], o["scales"]
    scores = [w["time"]*p[columns["latency_ms"]]/s["decision_time_ms"]
              + w["energy"]*p[columns["incremental_energy_j"]]/s["decision_energy_j"]
              + w["failure"]*p[columns["failure_probability"]] for p in predictions]
    candidates = [i for i in legal if predictions[i][columns["deadline_met"]]] or legal
    choice = min(candidates, key=lambda i: (scores[i], agent.contract.context["vms"][agent.contract.vm_ids[i]]["tier"] == "cloud", agent.contract.vm_ids[i]))
    agent.contract.validate_action(observation, choice)
    return choice, tuple(-x for x in scores)


class ObservedTeacher(PredictiveDQN):
    """Controller-compatible collector; online learned weights never choose actions."""
    def select(self, observation, epsilon=0):
        if epsilon != 0:
            raise ValueError("teacher is deterministic; collect without learning")
        return teacher_choice(self, observation)

    def controller(self, *, learn=False):
        if learn:
            raise ValueError("teacher rollouts must not update a DQN")
        return super().controller(learn=False)


def dataset_from_reports(reports):
    """Only actual legal decisions, including recovery; no forced reject labels."""
    result = []
    for report in reports:
        for t in report["rl"]["transitions"]:
            raw = t["state"]
            observation = EncodedObservation(raw["context_sha256"], raw["time_ms"], raw["task_id"],
                raw["placement_kind"], tuple(raw["values"]), tuple(raw["legal_mask"]))
            result.append((observation, t["action"]))
    return result


def agreement(agent, examples):
    result = {}
    for kind in ("all", "initial", "recovery"):
        group = [(s, a) for s, a in examples if kind == "all" or s.placement_kind == kind]
        correct = sum(agent.select(s, epsilon=0)[0] == a for s, a in group)
        result[kind] = dict(count=len(group), correct=correct, accuracy=correct/len(group) if group else None)
    return result


def behavior_clone(agent, examples, *, epochs=20, batch_size=64, learning_rate=.001):
    """Local seeded shuffles; reset only the *RL* optimizer/target after BC.

    Demonstration data never enter RL replay. BC optimizer/counters are separate;
    ordinary checkpoint format retains a complete resumable RL learner state.
    """
    for name, value in (("epochs", epochs), ("batch size", batch_size)):
        integer(value, name)
        if not value:
            raise ValueError("positive BC counts required")
    number("BC learning rate", learning_rate, positive=True)
    if agent.episode_active or agent.updates or agent.episodes or agent.decisions or agent.replay:
        raise ValueError("BC requires a fresh inactive learner")
    if not examples:
        raise ValueError("nonempty training-only teacher dataset required")
    for state, action in examples:
        agent.contract.validate_action(state, action)
        if action is None:
            raise ValueError("teacher labels require legal actions")
    before = network_digest(agent.online)
    before_agreement = agreement(agent, examples)
    x = torch.tensor([s.values for s, a in examples], dtype=torch.float32)
    mask = torch.tensor([s.legal_mask for s, a in examples], dtype=torch.bool)
    y = torch.tensor([a for s, a in examples], dtype=torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(agent.settings.seed)
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=learning_rate)
    rows = []
    for epoch in range(1, epochs+1):
        order = torch.randperm(len(examples), generator=generator)
        losses = []
        for indices in order.split(batch_size):
            logits = agent.online(x[indices]).masked_fill(~mask[indices], -torch.inf)
            loss = F.cross_entropy(logits, y[indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite masked BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.settings.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in agent.online.parameters()):
                raise ValueError("nonfinite BC weights")
            losses.append((len(indices), float(loss.detach())))
        rows.append(dict(epoch=epoch, cross_entropy=math.fsum(n*l for n, l in losses)/len(examples)))
    agent.target.load_state_dict(agent.online.state_dict())
    agent.optimizer = torch.optim.Adam(agent.online.parameters(), lr=agent.settings.learning_rate)
    return dict(kind="masked_behavior_cloning", epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, examples=len(examples), optimizer_updates=epochs*math.ceil(len(examples)/batch_size),
        initial_online_sha256=before, final_online_sha256=network_digest(agent.online),
        training_agreement_before=before_agreement, training_agreement_after=agreement(agent, examples),
        history=rows, rl_optimizer_reset=True, replay_empty=not agent.replay,
        scope="training-data agreement only; CE logits are not calibrated Q returns; no validation/test labels used")
