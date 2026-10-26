"""Basic CPU masked DQN for the enhanced finite-horizon scheduler (A-032).

A separate learner/checkpoint format; no Double DQN or benchmark training claim.
"""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from drlmots.enhanced_rl_contract import (EncodedObservation, EnhancedRLContract,
    RewardDelta, RewardSnapshot, reward_delta)
from drlmots.enhanced_telemetry import TELEMETRY_PROTOCOL
from drlmots.normalization import _digest
from drlmots.objective_model import integer, load_objective_model, number
from drlmots.run_config import _UniqueKeyLoader, load_run_config


@dataclass(frozen=True)
class EnhancedDQNConfig:
    seed: int = 42
    hidden_widths: tuple[int,int] = (64,64)
    learning_rate: float = .001
    replay_capacity: int = 2000
    batch_size: int = 32
    warmup_transitions: int = 32
    target_sync_updates: int = 50
    max_grad_norm: float = 10
    epsilon_start: float = 1
    epsilon_end: float = .05
    epsilon_decay_decisions: int = 2000
    telemetry_period_ms: float = 1
    recovery_enabled: bool = True

    def __post_init__(self):
        integer(self.seed,"DQN seed")
        widths=tuple(self.hidden_widths)
        if len(widths)!=2 or any(type(v) is not int or v<=0 for v in widths):
            raise ValueError("two positive hidden widths required")
        object.__setattr__(self,"hidden_widths",widths)
        for name in ("replay_capacity","batch_size","warmup_transitions","target_sync_updates","epsilon_decay_decisions"):
            integer(getattr(self,name),name)
            if not getattr(self,name):
                raise ValueError("positive DQN counts required")
        if not self.batch_size<=self.warmup_transitions<=self.replay_capacity:
            raise ValueError("batch <= warmup <= replay capacity required")
        for name in ("learning_rate","max_grad_norm","telemetry_period_ms"):
            number(name,getattr(self,name),positive=True)
        for name in ("epsilon_start","epsilon_end"):
            number(name,getattr(self,name))
        if not 0<=self.epsilon_end<=self.epsilon_start<=1 or type(self.recovery_enabled) is not bool:
            raise ValueError("invalid exploration/recovery settings")

    @classmethod
    def parse(cls,data):
        if not isinstance(data,dict) or set(data)!={f.name for f in fields(cls)}:
            raise ValueError("DQN configuration requires exactly its documented fields")
        return cls(**data)


@dataclass(frozen=True)
class EnhancedExperience:
    state: EncodedObservation
    action: int
    feedback: RewardDelta
    next_state: EncodedObservation | None


def _legal_actions(contract,observation):
    if type(observation) is not EncodedObservation or not isinstance(observation.values,tuple) or not isinstance(observation.legal_mask,tuple):
        raise ValueError("immutable enhanced observation required")
    number("observation time",observation.time_ms)
    if observation.time_ms>=contract.context["horizon_ms"] or observation.placement_kind not in ("initial","recovery") or not isinstance(observation.task_id,str) or not observation.task_id:
        raise ValueError("invalid decision observation metadata")
    legal=[i for i,v in enumerate(observation.legal_mask) if v]
    contract.validate_action(observation,legal[0] if legal else None)
    return legal


def validate_experience(contract,item):
    if type(item) is not EnhancedExperience or type(item.feedback) is not RewardDelta:
        raise ValueError("typed enhanced experience required")
    _legal_actions(contract,item.state)
    contract.validate_action(item.state,item.action)
    r=item.feedback
    number("reward start",r.start_ms)
    number("reward end",r.end_ms)
    if type(r.terminated) is not bool or type(r.bootstrap_discount) not in (int,float):
        raise ValueError("invalid reward terminal/discount flags")
    if r.context_sha256!=contract.sha256 or r.start_ms!=item.state.time_ms or not r.start_ms<=r.end_ms<=contract.context["horizon_ms"]:
        raise ValueError("experience reward context/time mismatch")
    if any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in (r.energy_reward,r.time_reward,r.failure_reward,r.service_penalty_reward)):
        raise ValueError("invalid enhanced cost reward components")
    if r.terminated:
        if item.next_state is not None or r.bootstrap_discount!=0 or r.end_ms!=contract.context["horizon_ms"]:
            raise ValueError("terminal experience cannot bootstrap")
    else:
        if item.next_state is None or r.bootstrap_discount!=1 or item.next_state.time_ms!=r.end_ms or not any(item.next_state.legal_mask):
            raise ValueError("nonterminal target needs a legal next action")
        _legal_actions(contract,item.next_state)


class EnhancedQNetwork(nn.Module):
    def __init__(self,contract,settings):
        super().__init__()
        self.context_sha256=contract.sha256
        n=len(contract.context["features"])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings.seed)
            a,b=settings.hidden_widths
            self.layers=nn.Sequential(nn.Linear(n,a,device="cpu",dtype=torch.float32),nn.ReLU(),
                nn.Linear(a,b,device="cpu",dtype=torch.float32),nn.ReLU(),nn.Linear(b,len(contract.vm_ids),device="cpu",dtype=torch.float32))

    def forward(self,values):
        return self.layers(values)


def masked_targets(target,experiences):
    if not experiences or any(x.state.context_sha256!=target.context_sha256 for x in experiences):
        raise ValueError("masked target context/batch mismatch")
    rewards=torch.tensor([x.feedback.total for x in experiences],dtype=torch.float32)
    indices=[i for i,x in enumerate(experiences) if not x.feedback.terminated]
    with torch.no_grad():
        if indices:
            states=torch.tensor([experiences[i].next_state.values for i in indices],dtype=torch.float32)
            masks=torch.tensor([experiences[i].next_state.legal_mask for i in indices],dtype=torch.bool)
            if not masks.any(dim=1).all():
                raise ValueError("cannot bootstrap an all-false mask")
            rewards[indices]+=target(states).masked_fill(~masks,-torch.inf).max(dim=1).values
    if not torch.isfinite(rewards).all():
        raise ValueError("nonfinite masked Bellman target")
    return rewards


class EnhancedDQN:
    def __init__(self,contract,settings=None,*,host_nodes=None):
        if type(contract) is not EnhancedRLContract:
            raise ValueError("contract-4 enhanced learner required; legacy contexts/checkpoints are incompatible")
        self.contract,self.settings=contract,settings or EnhancedDQNConfig()
        if type(self.settings) is not EnhancedDQNConfig:
            raise ValueError("enhanced DQN configuration required")
        c=contract.context
        inferred={v["host_id"]:v["node_id"] for v in c["vms"].values()}
        self.host_nodes=dict(sorted((inferred if host_nodes is None else host_nodes).items()))
        if set(self.host_nodes)!=set(c["hosts"]) or set(self.host_nodes.values())-set(c["failures"]) or any(self.host_nodes[v["host_id"]]!=v["node_id"] for v in c["vms"].values()):
            raise ValueError("host/node map must match topology; use from_configs for unused hosts")
        self.environment_sha256=_digest(dict(contract=contract.sha256,host_nodes=self.host_nodes,
            telemetry_protocol=TELEMETRY_PROTOCOL,period_ms=self.settings.telemetry_period_ms,recovery_enabled=self.settings.recovery_enabled))
        self.online=EnhancedQNetwork(contract,self.settings)
        self.target=deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.settings.learning_rate)
        self.replay=deque(maxlen=self.settings.replay_capacity)
        self.selector_rng=random.Random((self.settings.seed+1)%2**63)
        self.replay_rng=random.Random((self.settings.seed+2)%2**63)
        self.decisions=self.updates=self.episodes=0
        self.losses=[]
        self.episode_active=False

    @classmethod
    def from_configs(cls,run,model,settings=None,**contract_options):
        contract=EnhancedRLContract.from_configs(run,model,**contract_options)
        _,host_nodes=model.validate_topology(run)
        return cls(contract,settings,host_nodes=host_nodes)

    @property
    def epsilon(self):
        f=min(1,self.decisions/self.settings.epsilon_decay_decisions)
        return self.settings.epsilon_start+f*(self.settings.epsilon_end-self.settings.epsilon_start)

    def select(self,observation,epsilon=0):
        number("epsilon",epsilon)
        if epsilon>1:
            raise ValueError("epsilon exceeds one")
        legal=_legal_actions(self.contract,observation)
        if not legal:
            return None,()
        self.contract.validate_action(observation,legal[0])
        with torch.inference_mode():
            values=self.online(torch.tensor(observation.values,dtype=torch.float32))
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite online Q values")
        q=tuple(float(v) for v in values)
        action=self.selector_rng.choice(legal) if epsilon>0 and self.selector_rng.random()<epsilon else max(legal,key=lambda i:(q[i],-i))
        self.contract.validate_action(observation,action)
        return action,q

    def remember(self,item):
        validate_experience(self.contract,item)
        self.replay.append(item)
        if len(self.replay)<self.settings.warmup_transitions:
            return None
        batch=self.replay_rng.sample(tuple(self.replay),self.settings.batch_size)
        states=torch.tensor([x.state.values for x in batch],dtype=torch.float32)
        actions=torch.tensor([x.action for x in batch],dtype=torch.int64)
        targets=masked_targets(self.target,batch)
        predictions=self.online(states).gather(1,actions[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(predictions,targets)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),self.settings.max_grad_norm,error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.online.parameters()):
            raise ValueError("nonfinite DQN parameter")
        self.updates+=1
        self.losses.append(float(loss.detach()))
        if self.updates%self.settings.target_sync_updates==0:
            self.target.load_state_dict(self.online.state_dict())
        return self.losses[-1]

    def controller(self,*,learn=False):
        return EnhancedEpisodeController(self,learn=learn)

    def artifact(self):
        if self.episode_active:
            raise ValueError("save/fingerprint only between complete episodes")
        payload=dict(schema_version=1,kind="enhanced_masked_dqn_checkpoint",contract=self.contract.manifest(),
            environment_sha256=self.environment_sha256,host_nodes=self.host_nodes,settings=asdict(self.settings),
            online=_pack(self.online.state_dict()),target=_pack(self.target.state_dict()),optimizer=_pack(self.optimizer.state_dict()),
            replay=[asdict(x) for x in self.replay],selector_rng=self.selector_rng.getstate(),replay_rng=self.replay_rng.getstate(),
            decisions=self.decisions,updates=self.updates,episodes=self.episodes,losses=self.losses)
        return {**payload,"sha256":_digest(payload)}

    @property
    def fingerprint(self):
        return self.artifact()["sha256"]

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+".tmp")
        tmp.write_text(json.dumps(self.artifact(),allow_nan=False,indent=2),encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls,path,expected_contract,*,host_nodes=None):
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        checksum=data.pop("sha256",None)
        expected_keys=set("schema_version kind contract environment_sha256 host_nodes settings online target optimizer replay selector_rng replay_rng decisions updates episodes losses".split())
        if set(data)!=expected_keys or data.get("kind")!="enhanced_masked_dqn_checkpoint" or type(data.get("schema_version")) is not int or data.get("schema_version")!=1 or _digest(data)!=checksum:
            raise ValueError("invalid enhanced checkpoint kind/version/checksum")
        if data["contract"]!=expected_contract.manifest():
            raise ValueError("checkpoint observation/reward context mismatch")
        settings=EnhancedDQNConfig.parse(data["settings"])
        obj=cls(expected_contract,settings,host_nodes=host_nodes)
        if data["host_nodes"]!=obj.host_nodes or data["environment_sha256"]!=obj.environment_sha256:
            raise ValueError("checkpoint telemetry/environment mismatch")
        for name in ("online","target"):
            state=_unpack(data[name])
            expected=getattr(obj,name).state_dict()
            if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape or not torch.isfinite(state[k]).all() for k in state):
                raise ValueError("invalid checkpoint network")
            getattr(obj,name).load_state_dict(state,strict=True)
        optimizer_state=_unpack(data["optimizer"])
        if set(optimizer_state)!={"state","param_groups"} or _pack(optimizer_state["param_groups"])!=_pack(obj.optimizer.state_dict()["param_groups"]):
            raise ValueError("checkpoint optimizer settings differ from configuration")
        obj.optimizer.load_state_dict(optimizer_state)
        for name in ("decisions","updates","episodes"):
            integer(data[name],name)
            setattr(obj,name,data[name])
        parameter_ids=optimizer_state["param_groups"][0]["params"]
        if (obj.updates==0 and optimizer_state["state"]) or (obj.updates>0 and set(optimizer_state["state"])!=set(parameter_ids)):
            raise ValueError("checkpoint optimizer state/counters disagree")
        for param_id,param in zip(parameter_ids,obj.online.parameters()):
            if not obj.updates:
                break
            state=optimizer_state["state"][param_id]
            if set(state)!={"step","exp_avg","exp_avg_sq"} or state["step"].numel()!=1 or float(state["step"])!=obj.updates or any(state[k].shape!=param.shape or not torch.isfinite(state[k]).all() for k in ("exp_avg","exp_avg_sq")) or (state["exp_avg_sq"]<0).any():
                raise ValueError("invalid Adam checkpoint state")
        if len(data["replay"])>settings.replay_capacity or len(data["losses"])!=obj.updates:
            raise ValueError("invalid checkpoint counters/capacity")
        for value in data["losses"]:
            number("checkpoint loss",value)
        obj.losses=list(data["losses"])
        for raw in data["replay"]:
            item=EnhancedExperience(_observation(raw["state"]),raw["action"],RewardDelta(**raw["feedback"]),
                                    None if raw["next_state"] is None else _observation(raw["next_state"]))
            validate_experience(expected_contract,item)
            obj.replay.append(item)
        obj.selector_rng.setstate(_tuples(data["selector_rng"]))
        obj.replay_rng.setstate(_tuples(data["replay_rng"]))
        if obj.fingerprint!=checksum:
            raise ValueError("checkpoint round-trip differs")
        return obj


class EnhancedEpisodeController:
    def __init__(self,learner,*,learn=False):
        if type(learn) is not bool:
            raise ValueError("learn flag must be boolean")
        self.learner,self.contract,self.learn=learner,learner.contract,learn
        self.settings=learner.settings
        self.started=self.finished=False
        self.pending=None
        self.prefix=None
        self.records=[]
        self.transitions=[]

    def begin(self,expected_contract,host_nodes,recovery_enabled):
        if self.started or self.learner.episode_active or expected_contract.sha256!=self.contract.sha256 or dict(host_nodes)!=self.learner.host_nodes or recovery_enabled!=self.settings.recovery_enabled:
            raise ValueError("controller reuse or simulator/contract/environment mismatch")
        period=self.settings.telemetry_period_ms
        horizon=expected_contract.context["horizon_ms"]
        if horizon/period>1000000 or horizon+period==horizon:
            raise ValueError("telemetry period loses clock precision or exceeds one million polls")
        self.started=True
        self.learner.episode_active=True

    def _close(self,current,next_state=None):
        if self.pending is None:
            return
        obs,action,before=self.pending
        feedback=reward_delta(self.contract,before,current,terminated=next_state is None)
        item=EnhancedExperience(obs,action,feedback,next_state)
        validate_experience(self.contract,item)
        self.transitions.append(item)
        if self.learn:
            self.learner.remember(item)
        self.pending=None

    def __call__(self,observation,current):
        if not self.started or self.finished:
            raise ValueError("controller is outside its episode")
        legal=any(observation.legal_mask)
        if legal:
            self._close(current,observation)
            if self.prefix is None:
                self.prefix=reward_delta(self.contract,RewardSnapshot(),current)
        epsilon=self.learner.epsilon if self.learn else 0.0
        action,q=self.learner.select(observation,epsilon)
        if action is not None:
            self.pending=(observation,action,current)
            if self.learn:
                self.learner.decisions+=1
        self.records.append(dict(observation=asdict(observation),action=action,q_values=q,epsilon=epsilon))
        return action

    def finish(self,current):
        if not self.started or self.finished:
            raise ValueError("controller finish must occur exactly once")
        if self.pending is not None:
            self._close(current)
        if self.prefix is None:
            self.prefix=reward_delta(self.contract,RewardSnapshot(),current,terminated=True)
        expected=reward_delta(self.contract,RewardSnapshot(),current,terminated=True)
        actual=math.fsum([self.prefix.total]+[x.feedback.total for x in self.transitions])
        if not math.isclose(actual,expected.total,abs_tol=1e-9,rel_tol=1e-10):
            raise RuntimeError("episode reward did not reconcile")
        self.finished=True
        self.learner.episode_active=False
        if self.learn:
            self.learner.episodes+=1
        return dict(contract=self.contract.manifest(),environment_sha256=self.learner.environment_sha256,
            telemetry_protocol=TELEMETRY_PROTOCOL,telemetry_period_ms=self.settings.telemetry_period_ms,
            learn=self.learn,decisions=self.records,transitions=[asdict(x) for x in self.transitions],
            prefix_reward=asdict(self.prefix),terminal_reward_snapshot=asdict(current),
            total_reward=actual,expected_reward=expected.total,learnable_decisions=len(self.transitions))


def _pack(value):
    if isinstance(value,torch.Tensor):
        return {"tensor":value.tolist(),"dtype":str(value.dtype)}
    if isinstance(value,dict):
        return {str(k):_pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_pack(v) for v in value]
    return value


def _unpack(value):
    if isinstance(value,dict):
        if set(value)=={"tensor","dtype"}:
            if value["dtype"]!="torch.float32":
                raise ValueError("unsupported checkpoint tensor dtype")
            tensor=torch.tensor(value["tensor"],dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite checkpoint tensor")
            return tensor
        return {(int(k) if k.isdecimal() else k):_unpack(v) for k,v in value.items()}
    if isinstance(value,list):
        return [_unpack(v) for v in value]
    return value


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value,list) else value


def _observation(value):
    return EncodedObservation(value["context_sha256"],value["time_ms"],value["task_id"],value["placement_kind"],
                              tuple(value["values"]),tuple(value["legal_mask"]))


def run_enhanced_episode(learner,run,model,*,learn=False,**simulation):
    from drlmots.objective_sim import simulate_objectives
    before=learner.fingerprint if not learn else None
    controller=learner.controller(learn=learn)
    report=simulate_objectives(run,model,policy="masked_dqn",controller=controller,
        recovery_enabled=learner.settings.recovery_enabled,**simulation)
    if not learn and learner.fingerprint!=before:
        raise RuntimeError("frozen evaluation mutated the learner")
    report["rl"]["learner_updates"]=learner.updates
    report["rl"]["learner_decisions"]=learner.decisions
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description="Run enhanced masked-DQN placement; frozen inference, no training.")
    parser.add_argument("--config",default="configs/rl_network.yaml")
    parser.add_argument("--model-config",default="configs/objective_model.yaml")
    parser.add_argument("--dqn-config",default="configs/enhanced_dqn.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--workload-seed",type=int,default=6200)
    parser.add_argument("--failure-seed",type=int,default=7100)
    parser.add_argument("--output",default="outputs/objective-model/dqn-integration/inference.json")
    args=parser.parse_args(argv)
    try:
        run,model=load_run_config(args.config),load_objective_model(args.model_config)
        settings=EnhancedDQNConfig.parse(yaml.load(Path(args.dqn_config).read_text(encoding="utf-8-sig"),Loader=_UniqueKeyLoader))
        agent=EnhancedDQN.from_configs(run,model,settings)
        if args.checkpoint:
            agent=EnhancedDQN.load(args.checkpoint,agent.contract,host_nodes=agent.host_nodes)
            if agent.settings!=settings:
                raise ValueError("checkpoint settings differ from --dqn-config")
        report=run_enhanced_episode(agent,run,model,workload_seed=args.workload_seed,failure_seed=args.failure_seed)
        report["rl"]["inference_scope"]="loaded checkpoint" if args.checkpoint else "untrained random initialization; not a performance benchmark"
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        print(f"Saved {path}; {report['rl']['learnable_decisions']} placements; optimizer updates={agent.updates}; training disabled")
    except (ValueError,TypeError,OSError,KeyError) as error:
        parser.exit(2,f"enhanced DQN: {error}\n")


if __name__=="__main__":
    main()
