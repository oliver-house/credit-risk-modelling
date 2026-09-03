from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools.lock import _canonical

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock"


def _entries(path: Path) -> list[Requirement]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            lines.append(Requirement(line))
    return lines


@pytest.fixture(scope="module")
def declared() -> dict[str, Requirement]:
    return {_canonical(r.name): r for r in _entries(REQUIREMENTS)}


@pytest.fixture(scope="module")
def pinned() -> dict[str, str]:
    out = {}
    for requirement in _entries(LOCK):
        pins = [s for s in requirement.specifier if s.operator == "=="]
        assert len(pins) == 1, f"{requirement.name} is not pinned to one version"
        out[_canonical(requirement.name)] = pins[0].version
    return out


def test_the_lock_exists_and_is_populated(pinned):
    assert LOCK.exists()
    assert len(pinned) > 20, "the lock should hold the whole transitive closure"


def test_every_lock_entry_is_an_exact_pin(pinned):
    assert all(version for version in pinned.values())


def test_no_package_is_pinned_twice():
    names = [_canonical(r.name) for r in _entries(LOCK)]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"pinned more than once: {sorted(duplicates)}"


def test_every_declared_requirement_is_in_the_lock(declared, pinned):
    missing = sorted(set(declared) - set(pinned))
    assert not missing, (
        f"{missing} declared in requirements.txt but absent from requirements.lock. "
        f"Run `python tools/lock.py > requirements.lock`."
    )


def test_every_pin_satisfies_its_declared_range(declared, pinned):
    violations = [
        f"{name}=={pinned[name]} does not satisfy {requirement.specifier}"
        for name, requirement in declared.items()
        if name in pinned and not requirement.specifier.contains(pinned[name])
    ]
    assert not violations, (
        "requirements.lock is stale:\n  " + "\n  ".join(violations)
        + "\nRun `python tools/lock.py > requirements.lock`."
    )
