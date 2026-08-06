.PHONY: bootstrap test smoke configs full resume figures audit tables \
        manuscript-pilot manuscript-configs manuscript-runs manuscript-figures \
        manuscript-tables manuscript-audit manuscript

bootstrap:
	./scripts/bootstrap_env.sh

test:
	.venv/bin/python -m pytest -q

smoke:
	./scripts/run_smoke.sh

configs:
	./scripts/generate_full_configs.sh

full:
	./scripts/run_full.sh

resume:
	./scripts/resume_campaign.sh

figures:
	./scripts/make_all_figures.sh

audit:
	./scripts/audit_results.sh

tables:
	.venv/bin/python -m fr_gvi.experiments.tables

# --- manuscript campaign -------------------------------------------------
# Run once, in order: manuscript-pilot (selects and freezes the stepsizes),
# manuscript-configs, manuscript-runs, then figures, tables and audit.

manuscript-pilot:
	.venv/bin/python -m fr_gvi.experiments.manuscript --pilot
	.venv/bin/python -m fr_gvi.experiments.campaign configs/manuscript/pilot --budget-hours 0.5
	.venv/bin/python -m fr_gvi.experiments.pilot --write

manuscript-configs:
	.venv/bin/python -m fr_gvi.experiments.manuscript

manuscript-runs:
	./scripts/run_manuscript.sh

manuscript-figures:
	.venv/bin/python -m fr_gvi.plotting.manuscript_figures
	.venv/bin/python -m fr_gvi.plotting.audit --figures results/figures/manuscript

manuscript-tables:
	.venv/bin/python -m fr_gvi.experiments.manuscript_tables

manuscript-audit:
	.venv/bin/python -m fr_gvi.experiments.manuscript_audit

manuscript: manuscript-runs manuscript-figures manuscript-tables manuscript-audit
