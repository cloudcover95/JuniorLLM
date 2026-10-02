# JuniorOS tick

`python3 scripts/os_tick_prod.py`

Budget: 45 W host class, 12 W T4 class, 8 core names per pass, download 0, bind 127.0.0.1.
This is a cap, not a measured watt draw.
Mesh: ~/.juniorhome/gaia_mesh/os_tick.jsonl

## Deck on the overlay

JuniorDeck is a separate identity from Gaia. Controls: MX 0-15 note, pads 16-23 step, ADC 26-29 gamma/filter/clock/mix, 10.1 in DSI surface.
RP2040 FUNCSEL and AINSEL are the muxes. DMA is 12 channels, ADC DREQ 36. Not present on the Linux host. live false. flashed false. admit false.
