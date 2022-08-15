using GlobalSensitivity, QuasiMonteCarlo, OrdinaryDiffEq, Statistics
using Monkeypox
using Plots

url = "./data/timeseries-country-confirmed.csv"
country = "United States"
data_on, acc, cases, datatspan, datadate = datasource!(url, country)

N = 329500000.0 # population
θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
lb = [0.0001, 0.0001, 0, 0.0001, 0.0001, 0.0]
ub = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
prob = controlmonkeypoxprob!(N, θ, acc, pknown)

function sensimulate!(prob::ODEProblem, N, θ, datatspan, pknown)
    B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = θ[7]
    p0 = [B, μ, θ[1], θ[2], θ[3], θ[4], δ, θ[5], ϕ]
    u0 = [N - 1.0, 1.0, 0.0, θ[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob_pred_train = remake(prob, u0=u0, p=p0)
    sol = solve(prob_pred_train, Tsit5(), saveat=datatspan)
    return sol
end

f1 = function (p)
    datatspan = 0:length(acc)-1
    sol = sensimulate!(prob, N, p, datatspan, pknown)
    return maximum(sol[10, :])
end

bounds = [[0.000001, 1.0], [0.00001, 1.0], [0.00001, 0.2], [0.00001, 1.0], [0.00001, 1.0], [0.000001, 0.1], [0.00001, 1.0]]

reg_sens = gsa(f1, RegressionGSA(true), bounds, samples=1000)
bar(["pend", "p0", "r", "σ", "h", "α", "ϕ"],reg_sens.partial_correlation[1, :], label="PRCC")
savefig("./output/sensitivity.png")
