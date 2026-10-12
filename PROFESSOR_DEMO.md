# Professor demonstration guide - v0.2.1

This checkpoint presents a tested synthetic resource-aware scheduling simulator and
two simple placement rules. It is a foundation for the DRLMOTS implementation. There
is no trained DQN or claim of reproducing the paper's numbers yet.

## Generate and open

From the repository root after README's virtual-environment setup:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m drlmots.demo --config configs/resources.yaml --output-dir outputs/demos
Start-Process .\outputs\demos\drlmots-demo.html
```

The HTML is self-contained and works offline in a normal browser. No server, account,
GPU or external assets are required. Selectors switch between precomputed simulator
runs; changing a selector does not retrain an agent or rerun the simulation. Regenerate
after changing Python code or YAML. The command keeps the config's topology, units
and resource profile and replaces workload settings with the three named presets.

Outputs (ignored by Git; another contributor regenerates them):

- `outputs/demos/drlmots-demo.html`: interactive report with task outcomes/provenance.
- `outputs/demos/comparison.json`: all scenarios, paired raw reports and SVG content.
- `outputs/demos/figures/`: 15 SVG and 15 PNG figures for reuse in slides/reports.
- The HTML export button downloads the selected scenario's paired report JSON.

## Five-minute walkthrough

1. **Purpose (30 seconds).** Explain task placement across fog/cloud VMs under queue
   and resource constraints. Identify this as the simulator milestone before learning.
2. **Worked example (one minute).** Select Worked example. Each task needs 2 abstract
   CPU units and 4 GB. Cloud and fog-a fit; fog-b does not. Six tasks arrive together;
   each eligible VM takes one running and one waiting task. Four complete, two fail
   explicitly, and the successful makespan is 4 ms. Show two execution blocks per VM.
3. **Comparison (one minute).** Select Seeded workload. Least queue completes 96/100
   and rejects 4; first fit completes 91/100 and rejects 9. Both see the same seed,
   arrivals, lengths, capacities and queue limits. Switch Timeline policy to inspect
   placement and occupancy. These are one-seed software observations, not a general
   performance conclusion or a DRLMOTS-vs-paper-baseline experiment.
4. **Pressure (one minute).** Select Queue pressure: 25 simultaneous tasks, 4 completed
   and 21 failed for each policy. Open task outcomes to show explicit rejection reasons.
   Show that fog-b remains idle because it cannot satisfy the task demand.
5. **Evidence and next step (90 seconds).** Expand reproducibility; export JSON. Show
   the unit catalogue, input hashes, queue limits and all-resource-restored check.
   Explain the next deliverable: arrival-time state snapshots, stable VM action indices
   and legal action masks; reward choices and DQN training follow that verified contract.

## Explain these numbers accurately

| Metric | Meaning |
|---|---|
| Makespan | Latest successful finish from simulation time zero; includes arrival gaps |
| Average completion time | Successful task's finish minus arrival, averaged; includes waiting |
| Average waiting time (JSON) | Successful task's start minus arrival, averaged |
| VM occupancy | Sum of service intervals / successful makespan; not measured CPU utilization |
| Failed task | Recorded terminal rejection due to no resource fit or all feasible queues full |

For the seeded preset, makespan is approximately 155.13 ms for least queue and
154.57 ms for first fit; mean completion times are approximately 3.54 and 3.76 ms.
First fit rejects more work, so its slightly smaller makespan alone is not evidence
that it performs better. Both policies drain per-VM queues FCFS. Least queue chooses
minimum running-plus-waiting load with a VM-ID tie break; first fit chooses the minimum
eligible VM ID. They are synthetic comparators, not CNN/LSTM/GGCN paper implementations.

## Likely questions

- **Where is reinforcement learning?** It is the next milestone. The simulator,
  resource ledger and reproducible evaluation path are implemented; the state/reward
  and learning loop are still pending.
- **Why does a VM stay idle?** It lacks capacity for the declared task profile.
- **Why these milliseconds?** The fixture explicitly interprets 1000 bytes as 2 ms.
  CPU units affect admission, not speed. The source's unitless task length is unresolved.
- **Is this a real cloud deployment?** No. It uses SimPy, serial service per VM and
  zero transfer delay. Physical host budgets, energy/reliability and real datasets
  have not been implemented.
- **Which paper?** The repo targets DRLMOTS cloud/fog (DOI 10.1007/s10586-024-04712-z).
  The supplied DRL-MLS Kubernetes PDF is a different paper and its definitions are
  not merged into this implementation. Original-source audit remains open.

## Verification and pickup

Python verification: `python -m pytest -q` passes 378 tests with the optional demo
dependencies installed; `python -m pip check` is clean. Without Matplotlib, the figure
integration test skips and the core simulator remains usable. Timeline/comparison/
occupancy PNGs were visually inspected; SVG namespaces and task tooltips were checked.
`node scripts/check_demo_controls.cjs` verifies controls/tables/provenance/exports in
a DOM stub when Node.js is available. It does not verify actual browser layout.
The agent's browser tool disallowed local-file preview; manually open the HTML and
check desktop/mobile layout before a live presentation.

Read HANDOFF.md and NEXT_STEPS.md for the exact next task; claim ownership in
progress.md before editing. Keep scientific limitations visible in any presentation.
