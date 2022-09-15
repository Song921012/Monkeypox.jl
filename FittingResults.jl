##
using Monkeypox
using Optim
using Optimization
using Plots
using Turing
using StatsPlots
using BSON: @save, @load
using BSON
using AxisArrays
using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Dates
#color = colorant"#bcafcf"
palette(:Paired_10)
pyplot()
countryarray = ["United States", "Spain", "Germany", "United Kingdom", "France", "Brazil", "Canada", "Netherlands"]
gayratio = 0.02
poparray = gayratio*[329500000.0, 47350000.0, 83240000.0, 55980000.0, 67390000.0, 212600000.0, 38010000.0, 17440000.0]
url = "./data/timeseries-country-confirmed.csv";

##

## ACC to Daily
function computedaily!(acc)
    mid = zeros(length(acc))
    mid[2:end] = acc[1:end-1]
    pred_daily = acc - mid
    return pred_daily
end

## Plot similation results
function plotsim!(sim, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
    accmean = timeseries_steps_mean(sim)[10, :]
    dailymean = computedaily!(accmean)
    accup = timeseries_steps_quantile(sim, 0.95)[10, :]
    dailyup = computedaily!(accup)
    acclow = timeseries_steps_quantile(sim, 0.05)[10, :]
    dailylow = computedaily!(acclow)
    plot(datadate, accmean[datatspan], lw=3, linecolor=linecolor, color=color, ribbon=(acclow, accup), fillalpha=0.5, label="Predicted daily cases", legend_background_color=nothing, title=country)
    display(scatter!(datadate, acc[datatspan], markercolor=markercolor, label="Observed acculated cases"))
    savefig("./output/controlacc$i.png")
    plot(datadate, dailymean[datatspan], lw=3, linecolor=linecolor, color=color, ribbon=(dailylow, dailyup), fillalpha=0.5, label="Predicted daily cases", legend_background_color=nothing, title=country)
    display(scatter!(datadate, cases[datatspan], markercolor=markercolor, label="Observed daily cases"))
    savefig("./output/controldaily$i.png")
end

## Get the simulation of USA
function siminference!(i, country, color, linecolor, markercolor)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    N = poparray[i] # population
    θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
    @load "./output/chain$i.bson" chainout
    println(country, "data parameter:", chainout[2])
    chain_array = Array(chainout[1])
    prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)

    function prob_func(prob, i, repeat)
        B = pknown[1]
        μ = pknown[2]
        δ = pknown[3]
        ϕ = pknown[4]
        θ = chain_array[rand(1:2000), 2:7]
        p0 = [B, μ, θ[1], θ[2], θ[3], θ[4], δ, θ[5], ϕ]
        u0 = [N - 1.0, 1.0, 0.0, θ[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        prob_pred_train = remake(prob, u0=u0, p=p0)
    end
    ensemble_prob = EnsembleProblem(prob_pred, prob_func=prob_func)
    sim = solve(ensemble_prob, Tsit5(), saveat=0:length(acc)-1, EnsembleThreads(), trajectories=1000)
    plotsim!(sim, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
end

## Get the simulation of other countries
function simopt!(i, country, color, linecolor, markercolor)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    N = poparray[i] # population
    θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
    @load "./output/pmin$i.bson" p_min
    println(country, "data parameter:", p_min)
    prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)
    function prob_func(prob, i, repeat)
        B = pknown[1]
        μ = pknown[2]
        δ = pknown[3]
        ϕ = pknown[4]
        randnum = 1.0 + 0.2 * randn()
        θ = p_min
        p0 = [B, μ, randnum * θ[1], randnum * θ[2], randnum * θ[3], θ[4], δ, θ[5], ϕ]
        u0 = [N - 1.0, 1.0, 0.0, θ[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        prob_pred_train = remake(prob, u0=u0, p=p0)
    end
    ensemble_prob = EnsembleProblem(prob_pred, prob_func=prob_func)
    sim = solve(ensemble_prob, Tsit5(), saveat=0:length(acc)-1, EnsembleThreads(), trajectories=1000)
    plotsim!(sim, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
end

## Reproduction number
function brn(ρ, h, σ, μ, δ, ϕ)
    A1 = ρ*h*(σ+μ)*(2.0*σ+4.0*μ+3.0*δ)*(2.0*μ+ϕ+δ+σ)
    A2 = 2.0 * μ^2+(3.0*δ+σ+(1.0+h)*ρ+ϕ*h)*μ+δ^2+(ϕ*h*ρ+σ)*δ+ρ*h*(ϕ+σ)
    A3 = (2.0*μ+δ+ σ)^2
    brn= A1/(A2*A3)
    return brn
end


function ebrn!(i,url,country)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outdate = data_on.Date
    @load "./output/pmin$i.bson" p_min
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
    #B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    p0,pend,r,σ,h = @view p_min[1:5]
    effecbrn = t -> brn(pairrate!(t, p0, pend, r), h, σ, μ, δ, ϕ)
    tspan = collect(0.0:length(acc)-1)
    ebrn = effecbrn.(tspan)
    display(plot(outdate, ebrn, lw =3, label="Effective reproduction number",legend_background_color=nothing, title=country))
    savefig("./output/effectiveR$i.png")
    return ebrn
end

function levelbrn!(i,url,country,levels)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outdate = data_on.Date
    @load "./output/pmin$i.bson" p_min
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
    #B = pknown[1]
    μ = pknown[2]
    δ = pknown[3]
    ϕ = pknown[4]
    p0,pend,r,σ,h = @view p_min[1:5]
    # function effecbrnlevel!(x,y)
    #     effecbrnlevel = brn(x, h, y, μ, δ, ϕ)
    # end
    ρvalue = collect(0:0.01:1)
    #σvalue = collect(0.5:0.01:1)
    #levelset = [effecbrnlevel!(x,y) for x in 0:0.01:1, y in 0.5:0.01:1]
    #display(contour(ρvalue,σvalue,levelset',levels = levels, contour_labels=true))
    # ylabel!("ρ")
    # xlabel!("σ")
    # savefig("./output/levelsetR$i.png")
    brnarray = [brn(ρ, h, σ, μ, δ, ϕ) for ρ in ρvalue]
    indvalue = findfirst(x->x>1.0, brnarray)
    criticlevalue = ρvalue[indvalue]
    display(plot(ρvalue,brnarray,lw=3,label="Basic reproduction number function on ρ",legend_background_color=nothing, title=country, annotations = (indvalue, brnarray[indvalue], Plots.text("Critical ρ value: $criticlevalue", :left))))
    xlabel!("ρ")
    println(criticlevalue)
    savefig("./output/rhoR$i.png")
end

##
# Basic Reproduction number
brndata= zeros(length(countryarray))
levels = [0.5,1.0,1.5]
for (i, country) in enumerate(countryarray)
    brnarray = ebrn!(i,url,country)
    brndata[i]=brnarray[1]
    levelbrn!(i,url,country,levels)
end
brnframe=DataFrame(brn=brndata)
CSV.write("output/effectivebrn.csv",brnframe)

##




## 
# United States
color = :cornflowerblue
linecolor = :darkturquoise
markercolor = :darksalmon
i=1
country = "United States"
siminference!(i, country, color, linecolor, markercolor)

## 
# other 7 countries
for (i, country) in enumerate(countryarray)
    color = :cornflowerblue
    linecolor = :darkturquoise
    markercolor = :darksalmon
    if i>1
    simopt!(i, country, color, linecolor, markercolor)
    end
end


##