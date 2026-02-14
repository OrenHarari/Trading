#!/usr/bin/env python3
"""Monitor the optimizer process"""
import os, subprocess, time, json

out = []
# Check process
r = subprocess.run(['pgrep', '-la', 'optimizer_inc'], capture_output=True, text=True)
out.append(f"PROCESS: {r.stdout.strip() or 'NOT RUNNING'}")

# Check log
log_path = '/workspaces/Trading/opt_2h_log.txt'
if os.path.exists(log_path):
    mtime = os.path.getmtime(log_path)
    age = time.time() - mtime
    with open(log_path) as f:
        lines = f.readlines()
    out.append(f"LOG: {len(lines)} lines, last modified {age:.0f}s ago")
    for l in lines[-5:]:
        out.append(f"  {l.rstrip()}")
else:
    out.append("LOG: NOT FOUND")

# Check progress
prog_path = '/workspaces/Trading/optimizer_2h_progress.json'
if os.path.exists(prog_path):
    data = json.loads(open(prog_path).read())
    completed = data.get('completed', [])
    n_results = len(data.get('results', []))
    out.append(f"PROGRESS: completed={completed}, results={n_results}")
else:
    out.append("PROGRESS: NOT FOUND")

# Check results
res_path = '/workspaces/Trading/optimizer_2h_short_results.txt'
if os.path.exists(res_path):
    out.append(f"RESULTS: AVAILABLE ({os.path.getsize(res_path)} bytes)")
else:
    out.append("RESULTS: NOT YET")

with open('/workspaces/Trading/monitor_out.txt', 'w') as f:
    f.write('\n'.join(out))
