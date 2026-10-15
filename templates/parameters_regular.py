"""Starter parameters for an ordinary EoS table."""

general_parameters = {
    "EoS_metadata": "path/to/metadata.json",
    "GridAxes": {"e": 40, "nB": 40},
    "OutputFolder": "Output",
    "EoSName": "MyRegularEoS",
    "Execution": {"workers": 1},
    "Checkpoint": {"enabled": True, "resume": True},
    "Projection": {"enabled": True},
    "ExtraInfo": True,
    "Overwrite": "default",
}
