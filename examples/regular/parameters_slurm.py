"""Slurm-plan configuration for the synthetic regular example."""

general_parameters = {
    "EoS_metadata": "metadata.json",
    "OutputFolder": "../../example-output",
    "EoSName": "RegularSynthetic2D",
    "RunSpecification": "_slurm",
    "GridAxes": {"e": 5, "nB": 5},
    "Projection": {"enabled": True, "nearest_cell_starts": 2},
    "Execution": {"workers": 1, "chunk_size": 5, "worker_task_size": 2},
    "Checkpoint": {"enabled": True, "resume": True},
    "ExtraInfo": True,
    "Overwrite": "default",
}
