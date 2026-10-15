"""Serial synthetic BINODALS demonstration."""

general_parameters = {
    "EoS_metadata": "metadata.json",
    "OutputFolder": "../../example-output",
    "EoSName": "FirstOrderBinodalsSynthetic",
    "RunSpecification": "_serial",
    "GridAxes": {
        "e": {"count": 2, "lower": 0.13, "upper": 0.17},
        "nB": {"count": 2, "lower": 0.10, "upper": 0.40},
    },
    "Projection": {"enabled": False},
    "Execution": {"workers": 1, "chunk_size": 2},
    "Checkpoint": {"enabled": True, "resume": True},
    "Export": {"include_unusable": True},
    "ExtraInfo": True,
    "Overwrite": "default",
}
