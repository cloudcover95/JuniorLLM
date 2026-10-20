# Agent sync

Local only. No model pull.

ports/agent_sync.py reads the production summary and updates the production agent.
Writes ~/.juniorhome/os/agent_sync.json.

Home rails can surface whether the agent was updated and the keep status.

Additive. MIT.
