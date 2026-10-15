# Synthetic reward and transition contract - version 1

This v0.3.0a3 checkpoint implements A-024's causal reward and baseline transition
export. The paper's multi-objective reward, energy and reliability remain unresolved.
This does not implement an interactive agent, Q-network, replay or learning loop.

## Reward at an arrival

For a legal selected VM, use its pre-placement observation:

```
projected_wait_ms = remaining_running_service_ms + queued_service_ms
own_service_ms = explicit_byte_length * 2 / 1000
L = projected_wait_ms + own_service_ms
reward = S / (S + L)
```

S is an explicit positive scale in ms. The fixture chooses 10 ms. All-false action
masks receive `-P`; the fixture chooses penalty P=1. Both parameters are configurable,
finite and positive. Rewards are dimensionless. Admission lies in [0,1] (floating
underflow may produce zero); rejection is strictly negative. With S=10: a 2 ms task
starting now gets 5/6, the same task behind 2 ms work gets 5/7, and rejection gets -1.
The ratio is evaluated without forming potentially overflowing S+L or L/S.

The original contract-1 API is preserved. The end-to-end live contract-2 pipeline
uses the same explicit surrogate direction with causal upload-plus-queue timing,
forced-chain replay credit and no completion bonus. See [LIVE_DQN.md](LIVE_DQN.md)
for both standard/network-aware training, actual results and full checkpoint resume.
