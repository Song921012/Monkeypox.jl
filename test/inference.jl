##
using DataFrames
using CSV
using Plots
using TimeSeries
using Statistics
using Dates
using Monkeypox
using DifferentialEquations
using Turing
using Optimization
using OptimizationOptimJL
using NLopt
using SciMLSensitivity
url = "./data/timeseries-country-confirmed.csv"
country = "United States"
data_on, acc, cases, datatspan, datadate = datasource!(url, country)

##
N = 329500000.0 # population
θ = [0.3, 0.3, 0.7, 0.01]# ρ,σ,h,α
pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
lb = [0.0001, 0.0001, 0.0001, 0.0]
ub = [1.0, 1.0, 1.0, 0.1]
alg = Optim.BFGS()

##
prob_pred = monkeypoxprob!(N, θ, acc, pknown)
data_daily_to_learn = @view cases[datatspan]
data_to_learn = @view acc[datatspan]
function loss(θ)
    pred = simulate!(prob_pred, N, θ, datatspan,pknown)
    #mid = zeros(length(acc))
    #mid[2:end] =  pred[10, 1:end-1]
    #pred_daily =  pred[10, :] - mid
    #cases_learn = pred_daily[datatspan]
    #accpred = pred[10,:]
    #acc_learn =  accpred[datatspan]
    loss = sum(abs2, (log.(acc[datatspan]) .- log.(pred[10, :])))
    #+ sum(abs2, (log.(data_daily_to_learn .+ 1) .- log.(cases_learn .+ 1))) # + 1e-5*sum(sum.(abs, params(ann)))
end
println(loss(θ))
loss1 = OptimizationFunction(loss, Optimization.AutoZygote())
prob = OptimizationProblem(loss1, θ, lb=lb, ub=ub)
sol1 = Optimization.solve(prob, alg, maxiters=1000)
p_min = sol1.u
println(p_min)

##
#p_min = [mean(chain[:ρ]), mean(chain[:σ]), mean(chain[:h])]
datespan = data_on.Date[datatspan]
println("data parameter:", p_min)
θ_0 = p_min
prediction = train(θ_0)
scatter(datespan, acc, label="Training data")
display(plot!(datespan, prediction[10, :], label="Real accumulated cases"))

plot(datespan, prediction[5, :])