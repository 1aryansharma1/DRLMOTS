"""Frozen live scheduling with a saved predictive DQN checkpoint."""
import argparse
from pathlib import Path

from drlmots.enhanced_dqn import run_enhanced_episode
from drlmots.enhanced_rl_contract import EnhancedRLContract
from drlmots.enhanced_sanity import _write_json
from drlmots.objective_model import load_objective_model
from drlmots.predictive_dqn import PredictiveDQN
from drlmots.run_config import load_run_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl_network.yaml")
    parser.add_argument("--model-config", default="configs/objective_model.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workload-seed", type=int, default=85000)
    parser.add_argument("--failure-seed", type=int, default=95000)
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--output", default="outputs/objective-model/predictive-training/inference.json")
    args = parser.parse_args(argv)
    try:
        path = Path(args.output)
        if path.exists():
            raise ValueError("choose a fresh output path; inference does not overwrite evidence")
        run, model = load_run_config(args.config), load_objective_model(args.model_config)
        contract = EnhancedRLContract.from_configs(run, model)
        _, hosts = model.validate_topology(run)
        agent = PredictiveDQN.load(args.checkpoint, contract, host_nodes=hosts)
        kwargs = dict(workload_seed=args.workload_seed, failure_seed=args.failure_seed)
        if args.no_failures:
            kwargs["outages"] = []
        report = run_enhanced_episode(agent, run, model, **kwargs)
        report["rl"].update(predictive_features=agent.features, network_feature_count=agent.features["network_feature_count"],
                            checkpoint_fingerprint=agent.fingerprint, inference_scope="loaded predictive checkpoint; frozen epsilon=0")
        _write_json(path, report)
        print(f"Saved {path}; model seed={agent.settings.seed}; prior episodes={agent.episodes}; updates={agent.updates}; evaluation updates=0")
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"predictive inference: {error}\n")


if __name__ == "__main__":
    main()
