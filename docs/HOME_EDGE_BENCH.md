# JuniorHome mirrors quant.py, does not replace it

`adaptations/omega_cad/quant.py` and `ports/terraform.py` + `ports/flagstaff.py` stay source of truth.
JuniorHome `web3node/bitnet_orig.py` is the stdlib twin for the 45W node when mlx is off.
`scripts/flagstaff_prod.py` and `scripts/quant_bench.py` stay the LLM-side benches.
