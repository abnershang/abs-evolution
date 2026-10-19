"""Synthetic benchmark with an intentionally defective implementation.

An outcome is accepted only when at least one evidence record has the exact
requested goal, kind='acceptance', and accepted is the JSON boolean true.
Build, transport and implementation observations never establish acceptance.
The output is exactly {"accepted": <boolean>}. Input is one JSON object with
`goal` and a list of `records`. Do not use external services or filesystem data.
"""

import json
import sys

data = json.load(sys.stdin)
print(json.dumps({"accepted": bool(data.get("records"))}))
