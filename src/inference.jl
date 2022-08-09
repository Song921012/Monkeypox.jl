@doc raw"""
    monkeypoxinference(du,u,p,t)

Define classic Monkeypox compartment model. 

Parameters: (population, infection rate, recovery rate)

```math
\begin{align}\frac{dS(t)}{dt} =& B + \left( \mu + \sigma \right) \left( 2 \mathrm{SS}\left( t \right) + \mathrm{SI}\left( t \right) + \mathrm{SR}\left( t \right) \right) - \left( \mu + \rho \right) S\left( t \righ" ⋯ 1481 bytes ⋯ "sigma + 2 \mu \right) \mathrm{RR}\left( t \right) \\\frac{dH(t)}{dt} =& \frac{\rho \left( 1 - h \right) I\left( t \right) S\left( t \right)}{N} + \frac{2 h \rho I\left( t \right) S\left( t \right)}{N}\end{align}
```
"""
@model function fitmodel(data, prob1) # data should be a Vector
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    β ~ truncated(Normal(0.4,0.5),0,1)
    γ ~ truncated(Normal(0.1,0.001),0,1)
    p = [β,γ]
    prob = remake(prob1, p = p)
    predicted = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat=0:1:length(data)-1)

    for i = 1:length(predicted)
        data[i] ~ Normal(predicted[i][10], σ)
    end
end
# Online Learning
Time_learn  = 120:10:150
p_0 = [0.2,0.1]
for t_max = Time_learn
    tspan_learn = (0.0, t_max)
    prob_pred = ODEProblem(SIR_pred, u_0, tspan_learn, p_0)
    data_to_learn = data_acc[1:t_max+1]
    data_daily_to_learn = data_daily[1:t_max+1]
    function train(θ)
        prob_pred_train = remake(prob_pred, p = θ)
        Array(solve(prob_pred_train, Vern7(), abstol=1e-12, reltol=1e-12, saveat=0:1:t_max))
    end
    function loss(θ, p)
        pred = train(θ)
        mid = zeros(length(data_to_learn))
        mid[2:end] = pred[4,1:end - 1]
        pred_daily = pred[4,:] - mid
        sum(abs2, (log.(data_to_learn) .- log.(pred[4,:]))) + sum(abs2, (log.(data_daily_to_learn .+ 1) .- log.(pred_daily .+ 1))) # + 1e-5*sum(sum.(abs, params(ann)))
    end
    println(loss([0.3,0.1], p_0))
    lb = [0.0001,0.0001]
    ub = [1,1]
    using GalacticOptim:OptimizationProblem
    using Optim
    loss1 = OptimizationFunction(loss, GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(loss1, p_0, lb=lb, ub=ub)
    sol1 = GalacticOptim.solve(prob, NelderMead(), maxiters=1000)
    p_min = sol1.u
    println(p_min)
    Turing.setadbackend(:forwarddiff)
    model = fitSIR(data_to_learn, prob_pred)
    chain = sample(model, NUTS(.45), MCMCThreads(), 2000, 3, progress=false, init_theta = sol1.u)
    @save "SIR_chain_day$t_max.bason"
    p_min = [mean(chain[:β]),mean(chain[:γ])]
    println("$t_max data parameter:",p_min)
    p_0 = p_min
    tspan_predict = (0.0, 150)
    scatter(data_to_learn,label="Training data")
    plot!(data_acc,label="Real accumulated cases")
    prob_prediction = ODEProblem(SIR_pred, u_0, tspan_predict, p_min)
    data_prediction = Array(solve(prob_prediction, Tsit5(), saveat=1))
    display(plot!(data_prediction[4,:],label="Predicted accumulated cases",xlabel  = "Days after Feb 25", title = "Ontario's Accumulated  Cases Train by $t_max days data", lw=2))
    savefig("./Results_saving/SIR_Fit_Ontario_accumulated_cases_by$t_max.png")
    mid = zeros(length(data_acc))
    mid[2:end] = data_prediction[4,1:end - 1]
    pred_daily = data_prediction[4,:] - mid
    scatter(data_daily,label="Real accumulated cases")
    display(plot!(pred_daily,label="Predicted Daily cases", xlabel  = "Days after Feb 25", title = "Ontario's Daily Cases Train by $t_max days data", lw=2))
    savefig("./Results_saving/SIR_Fit_Ontario_daily_cases_by$t_max.png")
end