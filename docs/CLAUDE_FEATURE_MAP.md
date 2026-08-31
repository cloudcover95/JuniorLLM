# Claude-shipped features → JuniorCloud local rails (Aug 2026)

We copy *capabilities*, not vendor APIs. Everything below must run offline on BitNet / GGUF / MLX.

| Shipped (Claude Platform / Claude Code, Aug 2026) | JuniorCloud rail |
|---------------------------------------------------|------------------|
| Computer use multi-action turns + browser tool | **JuniorLookFrame / NavMesh / Sphere AR** — device actions stay local; no Anthropic computer_toolset |
| Skills API (versioned SKILL.md folders) | **JuniorSkillVault** — `obsidian/skills/` + `agent/skills/` |
| Files API (upload once, reference by id) | **JuniorFileLedger** — local paths + content hashes |
| Memory topics + sensitive-topic opt-in | **JuniorMemoryTopics** with `sensitive=false` default |
| Dynamic workflows / overnight subagents | **JuniorOvernightAgent** |
| Skill/plugin security scanning | **JuniorGuardrailPipeline** |
| Maker/checker + refute loops | **Enhanced TDA disagreement_gate** + FieldCore trust |
| Dispatch while away | Overnight queue `data/agent/overnight/` |

Models in the port registry are local/high-quant only unless a portal explicitly opts into a hosted API.
