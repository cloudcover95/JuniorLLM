---
name: route-port
description: Choose which Junior custom LLM port a slice belongs to.
---
# Route port

Call `ports.registry.pick` mentally:

| slice | port |
|-------|------|
| field/access/gym | JuniorBitNetFieldCore |
| durable work/checkpoint | JuniorAstra |
| astra-class / reason / qwen | JuniorAstraReason |
| safety/refuse | JuniorFable |
| chat/interactive | JuniorGemma4-4B |
| night drift | JuniorNightTernary |
| linux overlay / juniorctl | JuniorAstraReason or FieldCore |

Write `port:` on the commit. Wrong port is a gotcha — fix GOTCHAS.md if you mix Astra runtime vs Reason.
