from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.compile_sheet import compile_sheet
from junior_bitnet.math import absmax_act, absmean, binarize
from junior_bitnet.prove import prove

try:
    from adaptations.omega_cad.quant import i2s_pack
except Exception:
    def i2s_pack(trits: list[int]) -> bytes:
        return b""

__all__ = ["absmean", "absmax_act", "binarize", "bitlinear", "compile_sheet", "i2s_pack", "prove"]
