@doc raw"""
    monkeypoxpair!(du,u,p,t)

Define classic Monkeypox compartment model. 

Parameters: (population, infection rate, recovery rate)

```math
\begin{align}\frac{dS(t)}{dt} =& B + \left( \mu + \sigma \right) \left( 2 \mathrm{SS}\left( t \right) + \mathrm{SI}\left( t \right) + \mathrm{SR}\left( t \right) \right) - \left( \mu + \rho \right) S\left( t \righ" ⋯ 1481 bytes ⋯ "sigma + 2 \mu \right) \mathrm{RR}\left( t \right) \\\frac{dH(t)}{dt} =& \frac{\rho \left( 1 - h \right) I\left( t \right) S\left( t \right)}{N} + \frac{2 h \rho I\left( t \right) S\left( t \right)}{N}\end{align}
```
"""
function monkeypoxpair!(du, u, p, t)
    B, μ, ρ, σ, δ, h, ϕ = p
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

@doc raw"""
    monkeypoxsystem!(du,u,p,t)

Define classic Monkeypox compartment model using `ModellingToolkit.jl`. 
```math
\begin{align}\frac{dS(t)}{dt} =& B + \left( \mu + \sigma \right) \left( 2 \mathrm{SS}\left( t \right) + \mathrm{SI}\left( t \right) + \mathrm{SR}\left( t \right) \right) - \left( \mu + \rho \right) S\left( t \righ" ⋯ 1481 bytes ⋯ "sigma + 2 \mu \right) \mathrm{RR}\left( t \right) \\\frac{dH(t)}{dt} =& \frac{\rho \left( 1 - h \right) I\left( t \right) S\left( t \right)}{N} + \frac{2 h \rho I\left( t \right) S\left( t \right)}{N}\end{align}
```
"""
function monkeypoxsystem()
    @parameters B, μ, ρ, σ, δ, h, ϕ
    @variables t S(t) I(t) R(t) SS(t) SI(t) II(t) SR(t) IR(t) RR(t) H(t)
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
    return ode
end



@doc raw"""
    monkeypoxprob!(N, θ, acc, pknown)

generate monkeypox ode problem
"""
function monkeypoxprob!(N, θ, acc, pknown)
    B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    tspan = (0.0, length(acc))
    p0 = [B, μ, θ[1], θ[2], δ, θ[3], ϕ]
    u0 = [N - 1.0, 1.0, 0.0, θ[4] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob = ODEProblem(monkeypoxpair!, u0, tspan, p0)
    return prob
end


@doc raw"""
    simulate!(prob::ODEProblem,N, θ, datatspan,pknown)

solve monkeypox model
"""
function simulate!(prob::ODEProblem, N, θ, datatspan, pknown)
    B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    p0 = [B, μ, θ[1], θ[2], δ, θ[3], ϕ]
    u0 = [N - 1.0, 1.0, 0.0, θ[4] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob_pred_train = remake(prob, u0=u0, p=p0)
    sol = solve(prob_pred_train, Tsit5(), saveat=datatspan)
    return sol
end




@doc raw"""
    controlmonkeypoxpair!(du,u,p,t)

Define classic Monkeypox compartment model. 

Parameters: (population, infection rate, recovery rate)

```math
\begin{align}\frac{dS(t)}{dt} =& B + \left( \mu + \sigma \right) \left( 2 \mathrm{SS}\left( t \right) + \mathrm{SI}\left( t \right) + \mathrm{SR}\left( t \right) \right) - \left( \mu + \rho \right) S\left( t \righ" ⋯ 1481 bytes ⋯ "sigma + 2 \mu \right) \mathrm{RR}\left( t \right) \\\frac{dH(t)}{dt} =& \frac{\rho \left( 1 - h \right) I\left( t \right) S\left( t \right)}{N} + \frac{2 h \rho I\left( t \right) S\left( t \right)}{N}\end{align}
```
"""
function controlmonkeypoxpair!(du, u, p, t)
    B, μ, p0, pend, r, σ, δ, h, ϕ = p
    S, I, R, SS, SI, II, SR, IR, RR, H = u
    N = S + I + R
    ρ = pairrate!(t, p0, pend, r)
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




@doc raw"""
    controlmonkeypoxprob!(N, θ, acc, pknown)

generate monkeypox ode problem
"""
function controlmonkeypoxprob!(N, θ, acc, pknown)
    B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    tspan = (0.0, length(acc))
    p0 = [B, μ, θ[1], θ[2], θ[3], θ[4], δ, θ[5], ϕ]
    u0 = [N - 1.0, 1.0, 0.0, θ[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob = ODEProblem(controlmonkeypoxpair!, u0, tspan, p0)
    return prob
end


@doc raw"""
    controlsimulate!(prob::ODEProblem,N, θ, datatspan,pknown)

solve monkeypox model
"""
function controlsimulate!(prob::ODEProblem, N, θ, datatspan, pknown)
    B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    p0 = [B, μ, θ[1], θ[2], θ[3], θ[4], δ, θ[5], ϕ]
    u0 = [N - 1.0, 1.0, 0.0, θ[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob_pred_train = remake(prob, u0=u0, p=p0)
    sol = solve(prob_pred_train, Tsit5(), saveat=datatspan)
    return sol
end