"""Package the Blender add-on as an installable zip.

    python scripts/build_addon.py        ->  dist/blender_math_bridge.zip

Install it in Blender via Edit > Preferences > Add-ons > Install from Disk
(Blender 4.2+ also accepts it as an extension thanks to blender_manifest.toml).
"""

import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "blender_addon" / "blender_math_bridge"
OUT = ROOT / "dist" / "blender_math_bridge.zip"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, pathlib.Path("blender_math_bridge") / path.relative_to(SRC))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
