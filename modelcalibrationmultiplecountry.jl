##
using Monkeypox
using Optim
using Optimization
using Plots
using Turing
using StatsPlots
using BSON:@save,@load
countryarray = ["United States", "Spain", "Germany", "United Kingdom", "France", "Brazil", "Canada", "Netherlands"]
# 0.01
# Ca2.5
gayratio = 0.02
poparray = gayratio*[329500000.0, 47350000.0, 83240000.0, 55980000.0, 67390000.0, 212600000.0, 38010000.0, 17440000.0]
url = "./data/timeseries-country-confirmed.csv"

##
function runinference!(i, country)
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
    @save "./output/pmin$i.bson" p_min
    chainout = controlmonkeypoxinference!(N, p_min, acc, cases, datatspan, pknown, lb, ub)
    @save "./output/chain$i.bson" chainout
    println(country, "data parameter:", p_min)
    prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)
    prediction = controlsimulate!(prob_pred, N, p_min, datatspan, pknown)
    scatter(datadate, acc[datatspan], label="Observed data")
    display(plot!(datadate, prediction[10, :], label="Predicted accumulated cases", title=country))
    savefig("./output/controlacc$i.png")
    #plot(datadate, prediction[2, :])
    #display(plot(chainout[1]))
    datatspan1 = 0:length(acc)-1
    prediction1 = controlsimulate!(prob_pred, N, p_min, datatspan1, pknown)
    mid = zeros(length(acc))
    mid[2:end] = prediction1[10, 1:end-1]
    pred_daily = prediction1[10, :] - mid
    scatter(datadate, cases[datatspan], label="Observed data")
    display(plot!(datadate, pred_daily[datatspan], label="Predicted daily cases", title=country))
    savefig("./output/controldaily$i.png")
    #plot(datadate, prediction[2, :])
    display(plot(chainout[1]))
    savefig("./output/controlchain$i.png")
    return p_min
end

##
using DataFrames
using CSV
Paradata = DataFrame()
for i in 1:length(countryarray)
    country = countryarray[i]
    pmin = runinference!(i,country)
    Paradata.country = pmin
end

##
using AxisKeys
using BSON:@load
Paradata = DataFrame()
for i in 1:length(countryarray)
    country = countryarray[i]
    @load "./output/pmin$i.bson" p_min
    push!(p_min,1/p_min[1],1/p_min[2],1/p_min[5])
    Paradata[!,country] = p_min 
end
CSV.write("output/parameters.csv",Paradata)


