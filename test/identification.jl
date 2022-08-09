using Monkeypox
using ModelingToolkit
using StructuralIdentifiability
"""
@parameters B, μ, ρ, σ, δ, h, ϕ
@variables t S(t) I(t) R(t) SS(t) SI(t) II(t) SR(t) IR(t) RR(t) H(t) y1(t)
D = Differential(t)
eqs = [
    D(S) ~ B - (μ + ρ) * S + (σ + μ) * (2 * SS + SI + SR),
    D(I) ~ -(μ + ρ + δ) * I + (σ + μ) * (2 * II + SI + IR),
    D(R) ~ δ * I - (μ + ρ) * R + (σ + μ) * (2 * RR + SR + IR),
    D(SS) ~ 0.5 * ρ * S^2 / (S + I + R) - (σ + 2 * μ) * SS,
    D(SI) ~ ρ * (1 - h) * S * I / (S + I + R) - (σ + ϕ * h + 2 * μ + δ) * SI,
    D(II) ~ 0.5 * ρ * I^2 / (S + I + R) + ρ * h * S * I / (S + I + R) + ϕ * h * SI - (σ + 2 * μ + δ) * II,
    D(SR) ~ ρ * S * R / (S + I + R) + δ * SI - (σ + 2 * μ) * SR,
    D(IR) ~ ρ * I * R / (S + I + R) + δ * II - (σ + 2 * μ + δ) * IR,
    D(RR) ~ 0.5 * ρ * I^2 / (S + I + R) + δ * IR - (σ + 2 * μ) * RR,
    D(H) ~ 2 * ρ * h * S * I / (S + I + R) + ρ * (1 - h) * S * I / (S + I + R),
]
ode = ODESystem(eqs, t, name=:Monkeypoxmodel)
measured_quantities = [y1 ~ H]
@time global_id = assess_identifiability(ode, measured_quantities=measured_quantities)
"""
ode = @ODEmodel(
    S'(t) = B - (μ + ρ) * S(t) + (σ + μ) * (2 * SS(t) + SI(t) + SR(t)),
    I'(t) = -(μ + ρ + δ) * I(t) + (σ + μ) * (2 * II(t) + SI(t) + IR(t)),
    R'(t) = δ * I(t) - (μ + ρ) * R(t) + (σ + μ) * (2 * RR(t) + SR(t) + IR(t)),
    SS'(t) =  ρ * S(t)^2 / (2*(S(t) + I(t) + R(t))) - (σ + 2 * μ) * SS(t),
    SI'(t) = ρ * (1 - h) * S(t) * I(t) / (S(t) + I(t) + R(t)) - (σ + ϕ * h + 2 * μ + δ) * SI(t),
    II'(t) = ρ * I(t)^2 / (2*(S(t) + I(t) + R(t))) + ρ * h * S(t) * I(t) / (S(t) + I(t) + R(t)) + ϕ * h * SI(t) - (σ + 2 * μ + δ) * II(t),
    SR'(t) = ρ * S(t) * R(t) / (S(t) + I(t) + R(t)) + δ * SI(t) - (σ + 2 * μ) * SR(t),
    IR'(t) = ρ * I(t) * R(t) / (S(t) + I(t) + R(t)) + δ * II(t) - (σ + 2 * μ + δ) * IR(t),
    RR'(t) =  ρ * I(t)^2 / (2*(S(t) + I(t) + R(t))) + δ * IR(t) - (σ + 2 * μ) * RR(t),
    H'(t) = 2 * ρ * h * S(t) * I(t) / (S(t) + I(t) + R(t)) + ρ * (1 - h) * S(t) * I(t) / (S(t) + I(t) + R(t)),
    y(t) = H(t)
)
idresult = assess_identifiability(ode,[ρ, σ, h])