# Agent management

Local only. No model pull.

## Layer
ports/agent_mgr.py registers agents and stores last envelope, trit score, and handoff.
Writes ~/.juniorhome/os/agents.jsonl and agent_status.json.

## OSai
ports/osai_agents.py ensures junioros, goldend, and home are registered, runs a compute-smart tick for each, and writes ~/.juniorhome/os/osai_agents.json.

Home rails can surface which agent is on which layer without running the cycle again.

Additive. MIT.
