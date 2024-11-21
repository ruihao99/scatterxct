# My definition of canonical pulse function

In these notes, I document the procedure for consistently generating the Floquet Hamiltonian for any pulsed signal that can be decomposed into a cosine or sine function multiplied by an envelope function. Specifically, I will introduce the concept of a "canonical pulse function," which simplifies the process of constructing the Floquet Hamiltonian.

In the dipole approximation, the laser-matter interaction Hamiltonian is given by $H1 = - mu E(t)$, where mu denotes the dipole operator, and E(t) denotes the electric field (the pulse in this code base).

Regardless of the basis, the dipole operator itself is a Hermitian operator. To maintain the Hermiticity of the Hamiltonian, the electric field must be real. 

The common electric field functions include the continuous wave (CW) functions, such as the cosine, sine, and sine-squared functions, and the pulse functions, such as the Gaussian (Morlet) functions. The user generally want to define the pulse parameters in its natural "from", for example, the cosine pulse function will be given by $E(t) = E0 * cos(omega t + phi)$, where E0, omega, and phi are the amplitude, frequency, and phase of the pulse, respectively.

However, since the user would like to choose either sine function or cosine functions, and each the carrier function may include a phase phi. This will make the generating the Floquet Hamiltonian a bit tricky. To simplify the process, I define the canonical pulse function as
    $$E(t) = Epsilon(t) * e^{i omega t} + Epsilon^*(t) * e^{-i omega t}$$,
where Epsilon(t) is generally a complex function. Internally, the Floquet module will only use Epsilon(t) to generate the Floquet Hamiltonian. This approach can generalize the Floquet Hamiltonian generation process to any pulse function, including the cosine and sine functions.

Hence, if the user wants to extend the pulse classes. Not only do they want to define the natrual pulse function, but also the canonical pulse function.