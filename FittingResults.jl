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
    ?? = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ??,??,h,??
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,??,??,??
    @load "./output/chain$i.bson" chainout
    println(country, "data parameter:", chainout[2])
    chain_array = Array(chainout[1])
    prob_pred = controlmonkeypoxprob!(N, ??, acc, pknown)

    function prob_func(prob, i, repeat)
        B = pknown[1]
        ?? = pknown[2]
        ?? = pknown[3]
        ?? = pknown[4]
        ?? = chain_array[rand(1:2000), 2:7]
        p0 = [B, ??, ??[1], ??[2], ??[3], ??[4], ??, ??[5], ??]
        u0 = [N - 1.0, 1.0, 0.0, ??[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
    ?? = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ??,??,h,??
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,??,??,??
    @load "./output/pmin$i.bson" p_min
    println(country, "data parameter:", p_min)
    prob_pred = controlmonkeypoxprob!(N, ??, acc, pknown)
    function prob_func(prob, i, repeat)
        B = pknown[1]
        ?? = pknown[2]
        ?? = pknown[3]
        ?? = pknown[4]
        randnum = 1.0 + 0.2 * randn()
        ?? = p_min
        p0 = [B, ??, randnum * ??[1], randnum * ??[2], randnum * ??[3], ??[4], ??, ??[5], ??]
        u0 = [N - 1.0, 1.0, 0.0, ??[6] * N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        prob_pred_train = remake(prob, u0=u0, p=p0)
    end
    ensemble_prob = EnsembleProblem(prob_pred, prob_func=prob_func)
    sim = solve(ensemble_prob, Tsit5(), saveat=0:length(acc)-1, EnsembleThreads(), trajectories=1000)
    plotsim!(sim, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
end

## Reproduction number
function brn(??, h, ??, ??, ??, ??)
    A1 = ??*h*(??+??)*(2.0*??+4.0*??+3.0*??)*(2.0*??+??+??+??)
    A2 = 2.0 * ??^2+(3.0*??+??+(1.0+h)*??+??*h)*??+??^2+(??*h*??+??)*??+??*h*(??+??)
    A3 = (2.0*??+??+ ??)^2
    brn= A1/(A2*A3)
    return brn
end


function ebrn!(i,url,country)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outdate = data_on.Date
    @load "./output/pmin$i.bson" p_min
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,??,??,??
    #B = pknown[1]
    ?? = pknown[2]
    ?? = pknown[3]
    ?? = pknown[4]
    p0,pend,r,??,h = @view p_min[1:5]
    effecbrn = t -> brn(pairrate!(t, p0, pend, r), h, ??, ??, ??, ??)
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
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,??,??,??
    #B = pknown[1]
    ?? = pknown[2]
    ?? = pknown[3]
    ?? = pknown[4]
    p0,pend,r,??,h = @view p_min[1:5]
    # function effecbrnlevel!(x,y)
    #     effecbrnlevel = brn(x, h, y, ??, ??, ??)
    # end
    ??value = collect(0:0.01:1)
    #??value = collect(0.5:0.01:1)
    #levelset = [effecbrnlevel!(x,y) for x in 0:0.01:1, y in 0.5:0.01:1]
    #display(contour(??value,??value,levelset',levels = levels, contour_labels=true))
    # ylabel!("??")
    # xlabel!("??")
    # savefig("./output/levelsetR$i.png")
    brnarray = [brn(??, h, ??, ??, ??, ??) for ?? in ??value]
    indvalue = findfirst(x->x>1.0, brnarray)
    criticlevalue = ??value[indvalue]
    display(plot(??value,brnarray,lw=3,label="Basic reproduction number function on ??",legend_background_color=nothing, title=country, annotations = (indvalue, brnarray[indvalue], Plots.text("Critical ?? value: $criticlevalue", :left))))
    xlabel!("??")
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