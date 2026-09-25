# Quantum state rotation vs our emu

A qubit rotation is \(R_y(\theta)=\exp(-i\theta Y/2)\). On real amplitudes that is
\((a_0,a_1)\mapsto(\cos(\theta/2)a_0-\sin(\theta/2)a_1,\,\sin(\theta/2)a_0+\cos(\theta/2)a_1)\).
`ry01` does that on qutrit levels 0 and 1; level 2 is idle. Norm preserved on reals.

`cyclic` just permutes the three basis slots. That is a classical wheel, not a generator of SU(3).
Full qutrit rotations need eight Gell-Mann angles and complex phases. Not T0.
