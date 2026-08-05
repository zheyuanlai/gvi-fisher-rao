.PHONY: bootstrap test smoke configs full resume figures audit tables

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

