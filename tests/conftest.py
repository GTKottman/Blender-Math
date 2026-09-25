"""Test fixtures.

Tests that need Blender run the real add-on in-process through the ``bpy``
module (``pip install bpy``); they are skipped when it is not installed.
"""

from __future__ import annotations

import json

import pytest


class InProcessTransport:
    """Sends commands straight to the add-on's dispatcher (JSON round-trip included)."""

    def __init__(self, addon):
        self.addon = addon
        self.log = []

    def send(self, command, params=None):
        request = json.loads(json.dumps({"type": command, "params": params or {}}))
        response = json.loads(json.dumps(self.addon.dispatch(request)))
        self.log.append((command, response["status"]))
        if response["status"] != "ok":
            from blender_math_mcp.blender_client import BlenderError

            raise BlenderError(response["message"])
        return response["result"]


@pytest.fixture
def bpy_addon():
    bpy = pytest.importorskip("bpy")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import blender_math_bridge

    return blender_math_bridge


@pytest.fixture
def studio(bpy_addon):
    from blender_math_mcp.studio import MathStudio

    return MathStudio(InProcessTransport(bpy_addon))
