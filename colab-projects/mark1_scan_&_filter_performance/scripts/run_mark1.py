# scripts/run_mark1.py
from nbclient import NotebookClient
from nbformat import read, write
from pathlib import Path

nb_path = Path("colab-projects/mark1_scan_&_filter_performance/Mark1_Scan_&_Filter_performance.ipynb")
out_path = nb_path.with_suffix(".out.ipynb")

nb = read(nb_path.open(), as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
write(nb, out_path.open("w"))
print(f"wrote {out_path}")
