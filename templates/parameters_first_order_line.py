"""Starter parameters for a 2D LINE/Region-S EoS."""

general_parameters = {
    "EoS_metadata": "path/to/metadata_first_order_line.json",
    "GridAxes": {
        "e": {"count": 40, "lower": 0.001, "upper": 0.01},
        "nB": {"count": 40, "lower": 0.001, "upper": 0.02},
    },
    "OutputFolder": "Output",
    "EoSName": "MyLineEoS",
    "Execution": {"workers": 1},
    "Checkpoint": {"enabled": True, "resume": True},
    "Projection": {"enabled": False},
    "ExtraInfo": True,
    "Overwrite": "default",
}
