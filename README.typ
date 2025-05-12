#import "@preview/physica:0.9.5": *

#let Ld = $cal(L)$
#let dt = $d t$
#let psib = $overline(psi)$

== QED Field simulation
I am going to simulate the particle fields by taking the Euler-Lagrange equations for the fields, extracting the time-derivative parts from them to create the Newtonian mechanic time-evolution equations, and compute the fields in each increment of the timestamps using GPU.

Expanding the electron-photon field Lagrangian density:
$$$

Ld=i psib gamma^mu partial_mu psi - m psib psi - J^mu A_mu
- 1/4 F_(mu nu)F^(mu nu) \

= i psib gamma^mu partial_mu psi - m psib psi - q psib gamma^mu psi A_mu
- 1/4 (partial_mu A_nu - partial_nu A_mu)(partial^mu A^nu - partial^nu A^mu) 
$$$
Deriving the Euler-Lagrangian equation for $psi$:
$$$
0=partial_mu (partial Ld)/(partial(partial_mu psib))
- (partial Ld)/(partial psib) \

= 0 - (i gamma^mu partial_mu psi - m psi - q gamma^mu psi A_mu) \

-> i gamma^t partial_t psi = m psi+q gamma^mu psi A_mu - i vectorarrow(gamma) dot vectorarrow(nabla) psi \
"(multiply with" -i gamma^t ) \
-> partial_t psi = -i m gamma^t psi - i q gamma^t gamma^mu psi A_mu - gamma^t vectorarrow(gamma) dot vectorarrow(nabla) psi
$$$
As shown above, the field data I need to store is the $psi$ value for each point in space. In each timestamp, I will append the value $(partial_t psi dot dt)$ to $psi$, where $dt$ is the pseudo-infinitesimal timescale.

\
Next, doing the same for $A^mu$. The Euler-Lagrange equation for $A^mu$:
$$$
J^nu = partial_mu F^(mu nu) = partial_mu partial^mu A^nu
- partial^nu (partial_mu A^mu) \

= q psib gamma^nu psi \
= partial_t^2A^nu - laplacian A^nu - partial^nu (partial_t A^0 + div A) \
\
-> partial_t^2A^nu = q psib gamma^nu psi + laplacian A^nu
+ partial^nu (partial_t A^0 + div A)
$$$
Now consider individual case of $nu$-s:
$$$
partial_t^2A^t = q psib gamma^t psi + laplacian A^t
+ partial^t partial_t A^t + div (partial^t A)) \
\
-> 0 = q psib gamma^t psi + laplacian A^t + div (partial^t A))
$$$

$$$
partial_t^2 A^x = q psib gamma^x psi + laplacian A^x
+ partial^x partial_t A^t + partial^x div A \
\
-> 0 = q psib gamma^t psi + laplacian A^t + div (partial^t A))
$$$
As shown above, the $A^mu$ field data I need to store is the $A^mu$ and $partial_t A^mu$, where I append the value $(partial_t^2 A^mu dot dt)$ to $(partial_t A^mu)$ and $(partial_t A^mu dot dt)$ to $A^mu$ in each timestamp.
