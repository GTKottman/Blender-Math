"""Execute every ```python recipe in the blender-math skill against the real add-on.

Each line of a recipe block is a tool call ``tool(arg=value, ...)``.  It is run through the MCP
server (``mcp.call_tool``) with Blender running in-process via the ``bpy`` module, and any tool
error fails the check.  Keeps the skill's examples honest.

    python scripts/check_skill_recipes.py [--render OUT_DIR]
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "blender-math"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def recipe_blocks() -> list[tuple[str, int, str]]:
    out = []
    for md in sorted(SKILL.rglob("*.md")):
        text = md.read_text()
        for m in BLOCK.finditer(text):
            line = text[: m.start()].count("\n") + 2
            out.append((str(md.relative_to(ROOT)), line, m.group(1)))
    return out


def parse_calls(code: str) -> list[tuple[str, dict]]:
    calls = []
    for node in ast.parse(code).body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            raise ValueError(f"Recipe lines must be tool calls: {ast.unparse(node)}")
        call = node.value
        if call.args:
            raise ValueError(f"Use keyword arguments only: {ast.unparse(call)}")
        calls.append((call.func.id, {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}))
    return calls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", help="save a preview of each block to this directory")
    args = ap.parse_args(argv)

    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "blender_addon"), str(ROOT / "src")]
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import blender_math_bridge as addon
    from conftest import InProcessTransport

    from blender_math_mcp import server
    from blender_math_mcp.studio import Studio

    server._studio = Studio(InProcessTransport(addon))
    failures = 0

    async def run():
        nonlocal failures
        tools = {t.name for t in await server.mcp.list_tools()}
        for k, (path, line, code) in enumerate(recipe_blocks()):
            ok = True
            for name, kwargs in parse_calls(code):
                if name not in tools:
                    print(f"FAIL {path}:{line}: unknown tool {name}")
                    ok = False
                    break
                try:
                    res = await server.mcp.call_tool(name, kwargs)
                    err = res.is_error if hasattr(res, "is_error") else False
                    text = res.content[0].text if res.content and hasattr(res.content[0], "text") else ""
                except Exception as exc:  # noqa: BLE001 - MCP 1.x raises
                    err, text = True, str(exc)
                if err:
                    print(f"FAIL {path}:{line}: {name}({kwargs}) -> {text[:300]}")
                    ok = False
                    break
            if ok and args.render:
                out = pathlib.Path(args.render)
                out.mkdir(parents=True, exist_ok=True)
                img = server._studio.r.send("render", resolution=[800, 450])
                (out / f"{k:02d}_{pathlib.Path(path).stem}_{line}.png").write_bytes(
                    base64.b64decode(img["png_base64"]))
            print(("ok   " if ok else "FAIL ") + f"{path}:{line}")
            failures += not ok

    asyncio.run(run())
    print(f"{len(recipe_blocks()) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
