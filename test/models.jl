using DifferentialEquations
using Plots
using Monkeypox
using Sundials
##
## initial values and parameters
N = 329500000
u0 = [N - 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
tmax = 3650
tspan = (0, tmax)
B = 0
μ = 0
ρ = 1 / 15
σ = 1
δ = 1 / 30
h = 0.9
ϕ = 1
p0 = [B, μ, ρ, σ, δ, h, ϕ]
## run monkeypoxpair
odeprob = ODEProblem(monkeypoxpair, u0, tspan, p0)
sol = solve(odeprob, Vern7(), saveat=0:1:tmax)

plot(sol[2,:])

