#!/usr/bin/env python3
"""Deterministic JSON spectrum process used only by non-live integration tests."""

import json
import sys


request = json.load(sys.stdin)
protocol = request["protocol"]
geometry_hash = request["geometry_hash"]
energy = 1239.841984 / 650.0
json.dump(
    {
        "status": "SUCCESS",
        "states": [
            {
                "state_index": 1,
                "energy_ev": energy,
                "wavelength_nm": 650.0,
                "oscillator_strength": 0.2,
                "converged": True,
                "root_character": "pi-pi*",
            }
        ],
        "provenance": {
            "protocol": protocol,
            "geometry_hash": geometry_hash,
            "command_metadata": {"fixture": True},
            "backend_metadata": {"name": "fake-spectrum"},
        },
        "error": None,
    },
    sys.stdout,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
