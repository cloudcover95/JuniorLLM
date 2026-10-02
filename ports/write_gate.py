"""Envelope is closed only for t4 or host. Open is not a write."""
CLOSED = {"t4": {"files": 4, "chars": 256}, "host": {"files": 8, "chars": 1024}}
def closed(name):
    return name in CLOSED
