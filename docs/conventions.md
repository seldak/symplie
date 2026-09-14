# Conventions

## Attitude

\(R\in SO(3)\) maps body-frame coordinates into spatial-frame coordinates:

\[
    \mathbf{x}_{\mathrm{spatial}} = R\mathbf{x}_{\mathrm{body}}.
\]

The kinematic equation is

\[
    \dot R = R\widehat{\boldsymbol{\omega}},
\]

where angular velocity \(\boldsymbol{\omega}\), angular momentum
\(\boldsymbol{\pi}=J\boldsymbol{\omega}\), and applied torque
\(\boldsymbol{\tau}\) are expressed in body coordinates.

## Prescribed torque samples

`simulate_rigid_body` accepts one body-torque value at every state node. A
trajectory with `steps` transitions therefore requires an array with shape
`(steps + 1, 3)`. Transition \(k\) uses both
\(\boldsymbol{\tau}_k\) and \(\boldsymbol{\tau}_{k+1}\) in the discrete-force
update.

## Timesteps

The simulation functions accept either one scalar timestep or an array with
one positive interval per transition. For intervals
\(h_0,\ldots,h_{N-1}\), the state-node times are

\[
    t_0=0,
    \qquad
    t_k=\sum_{i=0}^{k-1}h_i.
\]

The timestep schedule is prescribed before simulation; it is not selected
adaptively from the evolving state.

## Feedback torque

`simulate_controlled_rigid_body` evaluates

\[
    \boldsymbol{\tau}_k =
    f_\tau(t_k,R_k,\boldsymbol{\pi}_k,J,p)
\]

at the beginning of each interval and holds it constant until the next sample.
For nonuniform timesteps, the evaluation time is the accumulated node time
\(t_k=\sum_{i<k}h_i\). The function returns one applied torque per transition,
with shape `(steps, 3)`. This models a digital controller under zero-order
hold; it is not the same sampling contract as a prescribed nodal torque
history.

## Twist coordinates

\(SE(3)\) twists use translation first:

\[
    \boldsymbol{\xi}
    =
    \begin{bmatrix}
        \boldsymbol{\rho} \\
        \boldsymbol{\phi}
    \end{bmatrix},
\]

where \(\boldsymbol{\rho}\) is the translational component and
\(\boldsymbol{\phi}\) is the rotation vector.

## Nonlinear-solver diagnostics

The Moser–Veselov solve uses a fixed Newton iteration budget. `converged`
reports whether the final residual meets the requested tolerance; it does not
mean that the loop terminated early. Simulation functions return trajectories
even when a step fails, so callers should check every convergence flag.
