function monkeypoxidentify()
    ode = @ODEmodel(
    S'(t) = B - (μ + ρ) * S(t) + (σ + μ) * (2 * SS(t) + SI(t) + SR(t)),
    I'(t) = -(μ + ρ + δ) * I(t) + (σ + μ) * (2 * II(t) + SI(t) + IR(t)),
    R'(t) = δ * I(t) - (μ + ρ) * R(t) + (σ + μ) * (2 * RR(t) + SR(t) + IR(t)),
    SS'(t) = 0.5 * ρ * S(t)^2 / (S(t)+I(t)+R(t)) - (σ + 2 * μ) * SS(t),
    SI'(t) = ρ * (1 - h) * S(t) * I(t) / (S(t)+I(t)+R(t)) - (σ + ϕ * h + 2 * μ + δ) * SI(t),
    II'(t) = 0.5 * ρ * I(t)^2 / (S(t)+I(t)+R(t)) + ρ * h * S(t) * I(t) / (S(t)+I(t)+R(t)) + ϕ * h * SI(t) - (σ + 2 * μ + δ) * II(t),
    SR'(t) = ρ * S(t) * R(t) / (S(t)+I(t)+R(t)) + δ * SI(t) - (σ + 2 * μ) * SR(t),
    IR'(t) = ρ * I(t) * R(t) / (S(t)+I(t)+R(t)) + δ * II(t) - (σ + 2 * μ + δ) * IR(t),
    RR'(t) = 0.5 * ρ * I(t)^2 / (S(t)+I(t)+R(t)) + δ * IR(t) - (σ + 2 * μ) * RR(t),
    H'(t) = 2 * ρ * h * S(t) * I(t) / (S(t)+I(t)+R(t)) + ρ * (1 - h) * S(t) * I(t) / (S(t)+I(t)+R(t)),
    y(t) = H(t)
    )
    assess_local_identifiability(ode)
end