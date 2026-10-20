# Labelled path

Local only. No model pull.

## Path
ports/labelled_path.py runs the full path, labels trit and combined scores as modelled, and emits an event.
Writes ~/.juniorhome/os/labelled_path.json.

## Assert
ports/labelled_assert.py runs the path and asserts that a layer event was emitted.
Writes ~/.juniorhome/os/labelled_assert.json.

Home rails can surface the labelled scores and the assert result.

Additive. MIT.
