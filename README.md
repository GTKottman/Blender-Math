# Blender-Math

Motion math for AI-driven animation in Blender.

## `blender-motion` skill

`.claude/skills/blender-motion/` is a Claude Code skill that teaches an agent
to animate Blender scenes through a Blender MCP server (any server that
exposes a Python code-execution tool, such as `execute_blender_code`):

- **Easing**: linear, smoothstep `3t²−2t³`, smootherstep `6t⁵−15t⁴+10t³`,
  power ease in/out, back/anticipate, and CSS-style cubic Bézier timing
  curves. Linear, smoothstep and cubic Bézier are written as exact,
  editable Blender Bézier keys; everything else is baked.
- **Springs**: underdamped, critically damped and overdamped solutions, with
  initial velocity for smooth hand-offs.
- **Paths**: a Bézier curve for *where* plus an easing on Follow Path
  progress for *when*.
- **Verification**: measures velocity, acceleration and jerk from the
  F-curves (or evaluated world positions) and flags hard starts, overshoot
  and kinks.

| file | purpose |
|------|---------|
| `SKILL.md` | workflow the agent follows |
| `references/math.md` | formulas, derivative tables, spring tuning, symptom → fix |
| `scripts/motion.py` | helper sent into Blender; registers as `blender_motion` |

To use it in another project, copy `.claude/skills/blender-motion/` into that
project's `.claude/skills/` (or `~/.claude/skills/`).

## Testing

```sh
python3 .claude/skills/blender-motion/scripts/motion.py   # math self-checks, no Blender needed
```

The Blender functions were tested against the `bpy` 4.2 LTS wheel
(`pip install bpy==4.2.*`, Python 3.11). The Blender 4.4+ layered-action
code path is written but not yet tested there.
