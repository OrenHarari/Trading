#!/usr/bin/env python3
import subprocess, os
result = subprocess.run(['pgrep', '-la', 'python'], capture_output=True, text=True)
lines = result.stdout.strip().split('\n') if result.stdout.strip() else ['NONE']
log_size = os.path.getsize('/workspaces/Trading/optimizer_2h_output.log') if os.path.exists('/workspaces/Trading/optimizer_2h_output.log') else 0
with open('/workspaces/Trading/status_check.txt', 'w') as f:
    f.write(f'LOG_SIZE_BYTES: {log_size}\n')
    f.write(f'PYTHON_PROCS: {len(lines)}\n')
    for l in lines:
        f.write(f'  {l}\n')
    # Also read last lines of log
    if os.path.exists('/workspaces/Trading/optimizer_2h_output.log'):
        with open('/workspaces/Trading/optimizer_2h_output.log') as lf:
            all_lines = lf.readlines()
            f.write(f'\nLOG_LINES: {len(all_lines)}\n')
            f.write('LAST 10 LINES:\n')
            for l in all_lines[-10:]:
                f.write(l)
    # Check results files
    for fname in ['optimizer_2h_short_results.txt', 'optimizer_4h_short_results.txt']:
        exists = os.path.exists(f'/workspaces/Trading/{fname}')
        f.write(f'\n{fname}: {"EXISTS" if exists else "NOT FOUND"}\n')
