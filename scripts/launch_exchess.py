#!/usr/bin/env python3
#
# Launch cutechess GUI with a specified EXchess executable registered as an engine.
# Run from the run directory:
#
#   python3 launch_exchess.py EXchess_v2026_03_01
#
# The engine is added/updated in cutechess's engines.json config, then
# the cutechess GUI is launched.  If no executable is specified, defaults
# to "EXchess".
#

import json
import os
import sys

run_dir  = os.path.dirname(os.path.abspath(__file__))
cutechess = os.path.normpath(os.path.join(run_dir, "../tools/cutechess-1.4.0/build/cutechess"))
engines_json = os.path.expanduser("~/.config/cutechess.com/engines.json")

exe_name = sys.argv[1] if len(sys.argv) > 1 else "EXchess"
exe_path = os.path.join(run_dir, exe_name) if not os.path.isabs(exe_name) else exe_name

if not os.path.isfile(exe_path):
    print(f"Error: executable not found: {exe_path}", file=sys.stderr)
    sys.exit(1)

engine_entry = {
    "name": exe_name,
    "command": exe_path,
    "workingDirectory": run_dir,
    "stderrFile": "",
    "protocol": "xboard",
    "timeoutScaleFactor": 1.0
}

# Load existing engines list, or start fresh
engines = []
if os.path.exists(engines_json):
    with open(engines_json) as f:
        try:
            engines = json.load(f)
        except json.JSONDecodeError:
            engines = []

# Update existing entry with this name, or append a new one
for i, e in enumerate(engines):
    if e.get("name") == exe_name:
        engines[i] = engine_entry
        break
else:
    engines.append(engine_entry)

with open(engines_json, "w") as f:
    json.dump(engines, f, indent=2)
    f.write("\n")

print(f"Registered '{exe_name}' in {engines_json}")
print(f"Launching cutechess...")

os.execv(cutechess, [cutechess])
