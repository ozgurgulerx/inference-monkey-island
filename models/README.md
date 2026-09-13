# Model workspaces

Part 01 covers the five core model families listed in Day 001. Numbered folders reflect my current study order, starting with Qwen3.8-27B. I can change that order as I go. Exact checkpoint identifiers and revisions will be recorded when selected for an experiment.

| Model / family | Workspace |
| --- | --- |
| Qwen3.8-27B | [01-qwen3.8-27b/](01-qwen3.8-27b/README.md) |
| GLM-4.7-Flash | [02-glm-4.7-flash/](02-glm-4.7-flash/README.md) |
| GLM-5.3-Flash | [03-glm-5.3-flash/](03-glm-5.3-flash/README.md) |
| DeepSeek lineage → V4.1 Flash | [04-deepseek-v4.1-flash/](04-deepseek-v4.1-flash/README.md) |
| Nemotron 3.5 Lightning | [05-nemotron-3.5-lightning/](05-nemotron-3.5-lightning/README.md) |

Each workspace has a `README.md` for the overview, study checklist and experiment index; `architecture.md` for authored architecture notes; and `experiments/` for model-specific configurations, raw results and findings. The authoring sections are empty. Create individual experiment folders when work begins, such as `experiments/experiment-001-subject/`.

Keep one canonical report per experiment and link to it from the relevant day. Record the exact checkpoint revision, runtime version, hardware/topology, precision, workload and configuration with each experiment. Keep reusable benchmark helpers separate from model-specific reports.

For a journal day under `days/day-NNN-YYMMDD-subject/`, a model link can be:

```markdown
[Qwen model workspace](../../models/01-qwen3.8-27b/README.md)
```

Model study documents live here in `inference-planet-arrakis/models/`, alongside the public `days/`. Guidance, methodology, career information, credentials and assessment records remain private outside this public repository. Check the contents of each model document before publishing; a model folder is not a place for private evaluator notes or scores.
