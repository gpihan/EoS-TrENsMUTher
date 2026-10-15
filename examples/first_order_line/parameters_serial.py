"""Serial synthetic LINE/Region-S demonstration."""

general_parameters = {
    "EoS_metadata": "metadata.json",
    "OutputFolder": "../../example-output",
    "EoSName": "FirstOrderLineSynthetic",
    "RunSpecification": "_serial",
    "GridAxes": {
        "e": {"count": 2, "lower": 1.35e-3, "upper": 1.45e-3},
        "nB": {"count": 2, "lower": 4.50e-3, "upper": 4.70e-3},
    },
    "Projection": {"enabled": False},
    "Execution": {"workers": 1, "chunk_size": 2},
    "Checkpoint": {"enabled": True, "resume": True},
    "Export": {"include_unusable": True},
    "ExtraInfo": True,
    "Overwrite": "default",
}
