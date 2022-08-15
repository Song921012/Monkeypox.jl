using Monkeypox
using Optim
using Optimization
using Plots
url = "./data/timeseries-country-confirmed.csv"
country = "United States"
data_on, acc, cases, datatspan, datadate = datasource!(url, country)

##
#N = 38010000.0
N = 329500000.0 # population
θ = [0.3, 0.3, 0.7, 0.01]# ρ,σ,h,α
pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
lb = [0.0001, 0.0001, 0.0001, 0.0]
ub = [1.0, 1.0, 1.0, 0.1]
alg = Optim.NelderMead()
p_min = monkeypoxopt!(N, θ, acc, cases, datatspan, pknown, lb, ub, alg)
println("data parameter:", p_min)
prob_pred = monkeypoxprob!(N, θ, acc, pknown)
prediction = simulate!(prob_pred, N, p_min, datatspan, pknown)
scatter(datadate, acc, label="Training data")
display(plot!(datadate, prediction[10, :], label="Real accumulated cases"))
plot(datadate, prediction[2, :])

##
N = 329500000.0
#N = 329500000.0 # population
θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
lb = [0.0001, 0.0001,0, 0.0001, 0.0001, 0.0]
ub = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
alg = Optim.NelderMead()
p_min = controlmonkeypoxopt!(N, θ, acc, cases, datatspan, pknown, lb, ub, alg)
println("data parameter:", p_min)
prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)
prediction = controlsimulate!(prob_pred, N, p_min, datatspan, pknown)
scatter(datadate, acc, label="Training data")
display(plot!(datadate, prediction[10, :], label="Real accumulated cases"))
plot(datadate, prediction[2, :])