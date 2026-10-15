# Professor demonstration guide - updated for v0.3.1

There are now separate reference-rule demos and an actual trained-policy report.
The standard mode remains; network-aware standard/fog-preferred agents were also
trained with live simulated placement. No exact paper-reproduction claim is made.

## Actual trained-policy presentation

Run the one-command benchmark in LIVE_DQN.md. Open
`outputs/training/professor/training-demo.html`, or extract `professor-demo.zip` and
open the HTML inside the extracted folder. It displays real completed runs without
JavaScript or a server. Each of three models trained for 20 episodes; full checkpoints,
loss/return logs and actual task traces are linked. Evaluation is greedy/frozen on
three held-out workload seeds. The fog variant's tier preference is an explicit rule;
DQN learns placement among its permitted VMs. Show the rule-only fog-first row too.

Explain the actual finding: trained models do not beat every heuristic after these
short runs. Three workload seeds are not three independent model initializations.
Separate zero-network and network comparisons; do not compare their timings as if
they shared the same environment. Research robustness is the next study, not a claim
supported by this demo. Browser layout still needs normal desktop/mobile preview.

## Generate and open

From the repository root after README's virtual-environment setup:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m drlmots.demo --config configs/resources.yaml --output-dir outputs/demos
Start-Process .\outputs\demos\drlmots-demo.html
```

Open **`outputs/demos/drlmots-demo.html`**, the generated report. The file
`src/drlmots/templates/demo.html` is its source template and has no simulation data.
Opening that source now shows a link to the generated report and the generation
command instead of an empty dashboard. If you previously opened the source, close
that tab and open the generated report above.

For the **network-aware, three-policy demo**, use `configs/fog_cloud.yaml` and
`--output-dir outputs/demos/fog-cloud`, then open the report in that subfolder.
It includes explicit uploads/propagation, defaults to fog-first completion, exports
21 PNG/21 SVG figures and shows **Fog/cloud decisions and transmission times**.
Read FOG_CLOUD_POLICY.md for the exact timing rule and input assumptions. Its scenario
counts differ from the original zero-network walkthrough below; never mix their results.
All three compared policies use identical network settings within each scenario.

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
   Explain the next research deliverable: independent model seeds and validation-controlled studies.
   Arrival-time snapshots, masks, normalization, synthetic rewards/transitions and an
   live trained CPU DQN are implemented in the separate trained report. Generate an optional
   `--capture-decisions` report to show six 47-feature observations and explicit masks.

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

- **Where is reinforcement learning?** The separate training report shows actual
  agent-controlled episodes, losses and checkpoints under LIVE_DQN.md. The simulator,
  resource ledger, reproducible evaluation and initial raw state/action trace are
  implemented, along with training-only normalization and a causal synthetic reward.
  The CPU network, legal selector, replay, optimizer/target updates and full resume
  now work. The figures in this older reference-rule demo remain baseline results;
  trained-policy results are clearly separated.
- **Why does a VM stay idle?** It lacks capacity for the declared task profile.
- **Why these milliseconds?** The fixture explicitly interprets 1000 bytes as 2 ms.
  CPU units affect admission, not speed. The source's unitless task length is unresolved.
- **Is this a real cloud deployment?** No. It uses SimPy, serial service per VM and
  zero transfer delay in the original mode; the network mode adds explicit simulated
  uploads/propagation. Physical host budgets, energy/reliability and real datasets
  have not been implemented.
- **Which paper?** The repo targets DRLMOTS cloud/fog (DOI 10.1007/s10586-024-04712-z).
  The supplied DRL-MLS Kubernetes PDF is a different paper and its definitions are
  not merged into this implementation. Original-source audit remains open.

## Verification and pickup

Current v0.3.1 verification: `python -m pytest -q` passes 677 tests on Python 3.12.14
with demo dependencies; `python -m pip check` is clean. The v0.2.1 demo checkpoint
originally passed 378 tests. Without Matplotlib, the figure
integration test skips and the core simulator remains usable. Timeline/comparison/
occupancy PNGs were visually inspected; SVG namespaces and task tooltips were checked.
`node scripts/check_demo_controls.cjs` verifies controls/tables/provenance/exports in
a DOM stub when Node.js is available. It does not verify actual browser layout.
The agent's browser tool disallowed local-file preview; manually open the HTML and
check desktop/mobile layout before a live presentation.

Read HANDOFF.md and NEXT_STEPS.md for the exact next task; claim ownership in
progress.md before editing. Keep scientific limitations visible in any presentation.
