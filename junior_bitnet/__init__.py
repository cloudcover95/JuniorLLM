from junior_bitnet.bitlinear import bitlinear
from junior_bitnet.compile_sheet import compile_sheet
from junior_bitnet.math import absmax_act, absmean, binarize, sign_alias
from junior_bitnet.prove import prove

# back-compat name
def sign(xs):
    return binarize(xs)

try:
    from adaptations.omega_cad.quant import i2s_pack
except Exception:
    from junior_bitnet.math import absmean as _a

    def i2s_pack(trits):
        return b""

__all__ = ["absmean", "absmax_act", "binarize", "bitlinear", "compile_sheet", "prove", "i2s_pack"]
