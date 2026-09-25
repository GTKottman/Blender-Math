"""The skill's recipes must be valid tool calls that succeed against the real add-on."""

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_skill_recipes as chk  # noqa: E402


def test_skill_frontmatter():
    text = (chk.SKILL / "SKILL.md").read_text()
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n---\n", text, re.S)
    assert m and m.group(1) == "blender-math" and len(m.group(2)) < 1024
    for ref in re.findall(r"\]\((references/[^)]+)\)", text):
        assert (chk.SKILL / ref).exists(), ref


def test_recipes_parse():
    blocks = chk.recipe_blocks()
    assert len(blocks) > 20
    for path, line, code in blocks:
        assert chk.parse_calls(code), f"{path}:{line}"


def test_recipes_run():
    pytest.importorskip("bpy")
    assert chk.main([]) == 0
