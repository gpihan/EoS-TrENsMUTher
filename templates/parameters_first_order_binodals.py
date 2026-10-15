"""Starter parameters for a 2D BINODALS EoS."""

general_parameters = {
    "EoS_metadata": "path/to/metadata_first_order_binodals.json",
    "GridAxes": {
        "e": {"count": 40, "lower": 0.1, "upper": 0.2},
        "nB": {"count": 40, "lower": 0.0, "upper": 0.5},
    },
    "OutputFolder": "Output",
    "EoSName": "MyBinodalsEoS",
    "Execution": {"workers": 1},
    "Checkpoint": {"enabled": True, "resume": True},
    "Projection": {"enabled": False},
    "ExtraInfo": True,
    "Overwrite": "default",
}
