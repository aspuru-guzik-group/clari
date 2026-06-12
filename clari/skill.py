from __future__ import annotations

from importlib import resources


def skill_text() -> str:
    return resources.files("clari").joinpath("inference", "SKILL.md").read_text()


def main() -> None:
    print(skill_text(), end="")
