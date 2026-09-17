"""The bounded subprocess that parses one hostile-until-proven-otherwise file.

Reads the file's bytes from stdin, writes one JSON object to stdout, and
touches nothing else: no network, no filesystem, no second file. Its parent
(:func:`caal.company.extraction.extract`) kills it on a wall clock, so a
parser that will not finish costs a timeout rather than a stuck upload.

Run as ``python -m caal.company.extract_worker <filename>``. The filename is
used only to choose the parser; it is never opened.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict


def main(argv: list[str]) -> int:
    from caal.company.extraction import apply_worker_limits, extract_in_process

    # Address space and CPU are capped here, before anything is parsed, rather
    # than through a preexec hook in a multithreaded parent.
    apply_worker_limits()
    filename = argv[1] if len(argv) > 1 else ""
    data = sys.stdin.buffer.read()
    result = extract_in_process(filename, data)
    payload = asdict(result)
    payload["blocks"] = [
        {"text": block.text, "location": block.location} for block in result.blocks
    ]
    payload["warnings"] = list(result.warnings)
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main(sys.argv))
