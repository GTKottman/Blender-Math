"""Visual themes.  ``dark`` (math-video look) is the default."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    text: str
    axes: str
    grid: str
    grid_minor: str
    palette: tuple[str, ...]
    fill_opacity: float = 0.35
    notes: dict = field(default_factory=dict)


THEMES = {
    "dark": Theme(
        "dark", background="#0e1116", text="#ffffff", axes="#d8dde6", grid="#2b3340", grid_minor="#1b2129",
        palette=("#58c4dd", "#ffff00", "#83c167", "#fc6255", "#9a72ac", "#5cd0b3", "#f0ac5f", "#d147bd"),
    ),
    "light": Theme(
        "light", background="#ffffff", text="#111111", axes="#222222", grid="#d9dde3", grid_minor="#eef0f3",
        palette=("#1f5fbf", "#d62728", "#2a8a3e", "#8e44ad", "#e67e22", "#16a085", "#b8860b", "#c2185b"),
        fill_opacity=0.25,
    ),
    "chalkboard": Theme(
        "chalkboard", background="#1f3a2e", text="#f4f1e8", axes="#efeadb", grid="#35584a", grid_minor="#2a4a3c",
        palette=("#ffe28a", "#9fd8ff", "#ff9e9e", "#b8f2a6", "#e4b8ff", "#ffc38a", "#8af0e0", "#ffffff"),
    ),
}

DEFAULT_THEME = "dark"


def get_theme(name: str | None) -> Theme:
    key = (name or DEFAULT_THEME).lower()
    if key not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; choose from {sorted(THEMES)}")
    return THEMES[key]
