# Production manager

Local only. No model pull.

ports/production_manager.py runs production with sync and then the goldend reader.
Writes ~/.juniorhome/os/production_manager.json with keep, synced, and goldend_keep.

Home rails can surface this single status.

Additive. MIT.
