@doc raw"""
    monkeypoxpair(du,u,p,t)

Define classic Monkeypox compartment model. 

Parameters: (population, infection rate, recovery rate)

```math
\begin{align}\frac{dS(t)}{dt} =& B + \left( \mu + \sigma \right) \left( 2 \mathrm{SS}\left( t \right) + \mathrm{SI}\left( t \right) + \mathrm{SR}\left( t \right) \right) - \left( \mu + \rho \right) S\left( t \righ" ⋯ 1481 bytes ⋯ "sigma + 2 \mu \right) \mathrm{RR}\left( t \right) \\\frac{dH(t)}{dt} =& \frac{\rho \left( 1 - h \right) I\left( t \right) S\left( t \right)}{N} + \frac{2 h \rho I\left( t \right) S\left( t \right)}{N}\end{align}
```
"""
function monkeypoxpair(du, u, p, t)
    B, μ, ρ, σ, δ, h, ϕ= p
    S, I, R, SS, SI, II, SR, IR, RR, H = u
    N = S + I + R
    du[1] = B - (μ + ρ) * S + (σ + μ) * (2 * SS + SI + SR)
    du[2] = -(μ + ρ + δ) * I + (σ + μ) * (2 * II + SI + IR)
    du[3] = δ * I - (μ + ρ) * R + (σ + μ) * (2 * RR + SR + IR)
    du[4] = 0.5 * ρ * S^2 / N - (σ + 2 * μ) * SS
    du[5] = ρ * (1 - h) * S * I / N - (σ + ϕ * h + 2 * μ + δ) * SI
    du[6] = 0.5 * ρ * I^2 / N + ρ * h * S * I / N + ϕ * h * SI - (σ + 2 * μ + δ) * II
    du[7] = ρ * S * R / N + δ * SI - (σ + 2 * μ) * SR
    du[8] = ρ * I * R / N + δ * II - (σ + 2 * μ + δ) * IR
    du[9] = 0.5 * ρ * I^2 / N + δ * IR - (σ + 2 * μ) * RR
    du[10] = 2 * ρ * h * S * I / N + ρ * (1 - h) * S * I / N
end