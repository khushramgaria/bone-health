import json
from pathlib import Path

metrics_dir = Path("metrics_output")

files = [
    "summary.json",
    "siglip_report.json", 
    "vit_report.json",
    "ensemble_report.json",
    "table_performance_comparison.json"
]

for filename in files:
    filepath = metrics_dir / filename
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
            print(f"\n{filename}:")
            print(json.dumps(data, indent=2)[:500])  # First 500 chars
    else:
        print(f"\n{filename}: NOT FOUND")
