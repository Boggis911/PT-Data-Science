# AI Agent Contract
## Repo map
- colab-projects/mark1_scan_&_filter_performance/…
- scripts/run_mark1.py  # entrypoint
## What to do
- Keep inputs in project folder; write small CSV/JSON to outputs/.
- Never commit large binaries or secrets.
## How to run
- Install: pip install -r requirements.txt
- Execute: python scripts/run_mark1.py
- Tests: pytest tests/ (smoke test must pass)
## Done criteria
- Notebook executes without errors.
- outputs/metrics.csv produced.
