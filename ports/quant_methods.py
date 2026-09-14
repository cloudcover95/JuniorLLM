METHODS = {
    "absmean-1.58": "Home bitnet_orig",
    "trit5": "5 trits/byte",
    "trit3": "shape+gamma+trit5",
    "jtr1": "24B header + 2-bit",
    "i2s-gguf": "GGUF envelope, not llama loadable",
    "int4-nibble": "signed -8..7",
    "q4k-shaped": "256 superblock, not ggml Q4_K_M",
}
def pick(name="absmean-1.58"):
    return METHODS.get(name, METHODS["absmean-1.58"])
