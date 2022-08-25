##
using Monkeypox
using Optim
using Optimization
using Plots
using Turing
using StatsPlots
using BSON: @save, @load
using BSON
countryarray = ["United States", "Spain", "Germany", "United Kingdom", "France", "Brazil", "Canada", "Netherlands"]
poparray = [329500000.0, 47350000.0, 83240000.0, 55980000.0, 67390000.0, 212600000.0, 38010000.0, 17440000.0]
url = "./data/timeseries-country-confirmed.csv";

##
i = 1
country = "United States"
data_on, acc, cases, datatspan, datadate = datasource!(url, country)
bar(datadate, acc[datatspan], label="Accumulated cases", title=country)
savefig("./output/accdata$i.png")
bar(datadate, cases[datatspan], label="Daily cases", title=country)
savefig("./output/dailydata$i.png")
N = poparray[i] # population
θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
lb = [0.0001, 0.0001, 0, 0.0001, 0.0001, 0.0]
ub = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
alg = Optim.NelderMead()
p_min = controlmonkeypoxopt!(N, θ, acc, cases, datatspan, pknown, lb, ub, alg)
chainout = controlmonkeypoxinference!(N, p_min, acc, cases, datatspan, pknown, lb, ub)
@save "./output/chain$i.bson" chainout
println(country, "data parameter:", chainout[2])


using PackageCompiler
 create_sysimage([:DifferentialEquations,:Turing,:Plots,:DataFrames,:Optimization], sysimage_path="JuliaSysimage.dll", precompile_execution_file="./test/inference.jl")
##
prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)
function prob_func(prob, i, repeat)
    B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    θ = chain_array[rand(1:2000), 1:6]
    p0 = [B, μ, θ[1], θ[2], θ[3], θ[4], δ, θ[5], ϕ]
    u0 = [N - 1.0, 1.0, 0.0, θ[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    prob_pred_train = remake(prob, u0=u0, p=p0)
end
ensemble_prob = EnsembleProblem(prob_pred, prob_func=prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=100)
plot(sim)
simm = EnsembleSummary(sim; quantiles=[0.05, 0.95])
display(plot(simm, error_style=:bars))
display(plot(simm, fillalpha=0.3))
plot(simm, idxs=[2, 3], fillalpha=0.3)
xlabel!("days")
display(ylabel!("Case"))
