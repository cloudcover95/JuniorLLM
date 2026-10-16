CC ?= cc
CFLAGS ?= -O2 -fPIC -Wall
.PHONY: all clean so
all: so
so: libjunior_absmean.so libjunior_winsor.so libjunior_bitplane.so
libjunior_absmean.so: absmean.c
	$(CC) $(CFLAGS) -shared -lm absmean.c -o $@
libjunior_winsor.so: winsor.c
	$(CC) $(CFLAGS) -shared -lm winsor.c -o $@
libjunior_bitplane.so: bitplane.c
	$(CC) $(CFLAGS) -shared bitplane.c -o $@
clean:
	rm -f libjunior_*.so trit.spv
spirv:
	@command -v glslangValidator >/dev/null && glslangValidator -V asahi/trit.comp -o asahi/trit.spv || echo no-glslang
