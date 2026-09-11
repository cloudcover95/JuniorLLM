# JuniorTeqp vs NIST teqp

NIST teqp: Templated Equation of State Package — Helmholtz EOS, corresponding states, no handwritten derivatives (AD).

JuniorTeqp: same *shape* on trit vectors. Not REFPROP. Not fluids.

| teqp | JuniorTeqp |
|------|------------|
| T, rho | T = night/IQ drift scale, rho = nonzero fraction |
| T_c, rho_c | 0.22, 0.5 (AbsMean / review threshold) |
| A_id + A_res | trit entropy + neighbor coupling |
| get_Ar01 | finite-diff scale of residual |
| cubic / SAFT / GERG | FieldCore embed, night state, zk pack, AbsMean W_q |
| C++ AD | stdlib central difference |
