import sys
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent

HEADER = """\
# Exact pins for the container image, so a build is reproducible rather than
# whatever the index happened to serve that day. requirements.txt keeps its
# ranges for development; this is the transitive closure of those ranges as
# resolved on Python 3.14.7, with the dev tools left out.
#
# Regenerate after changing requirements.txt:
#   python -m pip install -r requirements.txt
#   python tools/lock.py > requirements.lock
"""


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def closure(roots: list[str], installed: dict) -> set[str]:
    """Every installed distribution reachable from the named roots."""
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        name = _canonical(queue.pop())
        if name in seen or name not in installed:
            continue
        seen.add(name)
        for requirement in installed[name].requires or []:
            parsed = Requirement(requirement)
            if parsed.marker is not None and not parsed.marker.evaluate({"extra": ""}):
                continue
            queue.append(parsed.name)
    return seen


def main() -> int:
    roots = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            roots.append(Requirement(line).name)

    installed = {_canonical(d.metadata["Name"]): d for d in distributions()}
    missing = [r for r in roots if _canonical(r) not in installed]
    if missing:
        print(f"Not installed in this environment: {missing}. "
              f"Run `pip install -r requirements.txt` first.", file=sys.stderr)
        return 1

    names = sorted(closure(roots, installed))
    print(HEADER)
    for name in names:
        dist = installed[name]
        print(f"{dist.metadata['Name']}=={dist.version}")
    print(f"Wrote {len(names)} pins", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
