# User utilities

`quickstart_api.py` is a compact typed-Python example that runs the bundled
synthetic table, saves an `inverse.eosinv` artifact, reloads it, and prints a
JSON status summary. It is useful when embedding the package in Python:

```bash
python utilities/quickstart_api.py --output quickstart-api-output
python utilities/quickstart_api.py --projection --output quickstart-projected-output
```

The input is non-physical and intended only for workflow checks.

`check_boundaries.py` validates a metadata/table pair and prints the automatic
target and/or tilde envelopes:

```bash
python utilities/check_boundaries.py path/to/metadata.json --space both
```
