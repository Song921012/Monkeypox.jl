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
gayratio = 0.02
countryarray = ["United States", "Spain", "Germany", "United Kingdom", "France", "Brazil", "Canada", "Netherlands"]
poparray = gayratio*[329500000.0, 47350000.0, 83240000.0, 55980000.0, 67390000.0, 212600000.0, 38010000.0, 17440000.0]
url = "./data/timeseries-country-confirmed.csv";
function computedaily!(acc)
    mid = zeros(length(acc))
    mid[2:end] = acc[1:end-1]
    pred_daily = acc - mid
    return pred_daily
end
function computedailycontrol!(acc)
    pred_daily = acc[2:end] - acc[1:end-1]
    return pred_daily
end
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
function dateall!(preddays, data_on)
    dateall = data_on.Date[1]:Day(1):(data_on.Date[end]+Day(preddays))
    return dateall
end
function datepredict!(preddays, data_on)
    datepredict = data_on.Date[end]:Day(1):(data_on.Date[end]+Day(preddays))
    return datepredict
end
function controlplotsim!(sim, controlsim, preddays, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
    accmean = [timeseries_steps_mean(sim)[10, :]; timeseries_steps_mean(controlsim)[10, 2:end]]
    dailymean = computedaily!(accmean)
    accup = [timeseries_steps_quantile(sim, 0.95)[10, :]; timeseries_steps_quantile(controlsim, 0.95)[10, 2:end]]
    dailyup = computedaily!(accup)
    acclow = [timeseries_steps_quantile(sim, 0.05)[10, :]; timeseries_steps_quantile(controlsim, 0.05)[10, 2:end]]
    dailylow = computedaily!(acclow)
    dateall = dateall!(preddays, data_on)
    plot(dateall, accmean, lw=3, linecolor=linecolor, color=color, ribbon=(acclow, accup), fillalpha=0.3, label="Predicted daily cases", legend_background_color=nothing, title=country)
    display(scatter!(datadate, acc[datatspan], markercolor=markercolor, label="Observed acculated cases"))
    savefig("./output/senocontrolacc$i.png")
    plot(dateall, dailymean, lw=3, linecolor=linecolor, color=color, ribbon=(dailylow, dailyup), fillalpha=0.3, label="Predicted daily cases", legend_background_color=nothing, title=country)
    display(scatter!(datadate, cases[datatspan], markercolor=markercolor, label="Observed daily cases"))
    savefig("./output/senocontroldaily$i.png")
end
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
## plotdata
function plotdata!(url, country, markercolor)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    p1 = scatter(datadate, acc[datatspan], markercolor=markercolor, label="Observed acculated cases", legend_background_color=nothing, title=country)
    p2 = scatter(datadate, cases[datatspan], markercolor=markercolor, label="Observed daily cases", legend_background_color=nothing, title=country)
    return p1, p2
end
## obtain sim
function solinference!(i, url, country)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outdate = data_on.Date
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
    return sim, outdate
end

## Obtain controlsim
function solcontrolinference!(sim, i, url, country, preddays, controlρ, controlσ, controlh, controlSS)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outputdate = datepredict!(preddays, data_on)
    N = poparray[i] # population
    θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
    @load "./output/chain$i.bson" chainout
    println(country, "data parameter:", chainout[2])
    chain_array = Array(chainout[1])
    prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)
    controlu0 = timeseries_steps_mean(sim)[:, end]
    function control_prob_func(prob, i, repeat)
        B = pknown[1]
        μ = pknown[2]
        δ = pknown[3]
        ϕ = pknown[4]
        θ = chain_array[rand(1:2000), 2:7]
        p0 = [B, μ, controlρ * θ[1], controlρ * θ[2], θ[3], controlσ *θ[4], δ, controlh * θ[5], ϕ]
        u0 = copy(controlu0)
        u0[4] = controlSS * u0[4]
        prob_pred_train = remake(prob, u0=u0, p=p0, tspan=(length(acc) - 1, length(acc) - 1 + preddays))
    end
    control_ensemble_prob = EnsembleProblem(prob_pred, prob_func=control_prob_func)
    controlsim = solve(control_ensemble_prob, Tsit5(), saveat=(length(acc)-1):(length(acc)-1+preddays), EnsembleThreads(), trajectories=1000)
    return controlsim, outputdate, controlρ, controlσ, controlh, controlSS
end

## integrated
function controlsiminference!(i, url, country, color, linecolor, markercolor, preddays, controlρ, controlσ, controlh, controlSS)
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
    controlu0 = timeseries_steps_mean(sim)[:, end]
    function control_prob_func(prob, i, repeat)
        B = pknown[1]
        μ = pknown[2]
        δ = pknown[3]
        ϕ = pknown[4]
        θ = chain_array[rand(1:2000), 2:7]
        p0 = [B, μ, controlρ * θ[1], controlρ * θ[2], θ[3], controlσ *θ[4], δ, controlh * θ[5], ϕ]
        u0 = copy(controlu0)
        u0[4] = controlSS * u0[4]
        prob_pred_train = remake(prob, u0=u0, p=p0, tspan=(length(acc) - 1, length(acc) - 1 + preddays))
    end
    control_ensemble_prob = EnsembleProblem(prob_pred, prob_func=control_prob_func)
    controlsim = solve(control_ensemble_prob, Tsit5(), saveat=(length(acc)-1):(length(acc)-1+preddays), EnsembleThreads(), trajectories=1000)
    controlplotsim!(sim, controlsim, preddays, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
end
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

## obtain sim opt
function solopt!(i, url,country)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outdate = data_on.Date
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
    return sim, outdate
end

## Obtain controlsim, opt version
function solcontrolopt!(sim, i, url, country, preddays, controlρ, controlσ, controlh, controlSS)
    data_on, acc, cases, datatspan, datadate = datasource!(url, country)
    outputdate = datepredict!(preddays, data_on)
    N = poparray[i] # population
    θ = [0.3, 0.3, 0.2, 0.1, 0.7, 0.01]# ρ,σ,h,α
    pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
    @load "./output/pmin$i.bson" p_min
    println(country, "data parameter:", p_min)
    prob_pred = controlmonkeypoxprob!(N, θ, acc, pknown)
    controlu0 = timeseries_steps_mean(sim)[:, end]
    function control_prob_func(prob, i, repeat)
        B = pknown[1]
        μ = pknown[2]
        δ = pknown[3]
        ϕ = pknown[4]
        randnum = 1.0 + 0.2 * randn()
        θ = p_min
        p0 = [B, μ, randnum*controlρ * θ[1], randnum*controlρ * θ[2], randnum * θ[3], controlσ*θ[4], δ, controlh * θ[5], ϕ]
        u0 = copy(controlu0)
        u0[4] = controlSS * u0[4]
        prob_pred_train = remake(prob, u0=u0, p=p0, tspan=(length(acc) - 1, length(acc) - 1 + preddays))
    end
    control_ensemble_prob = EnsembleProblem(prob_pred, prob_func=control_prob_func)
    controlsim = solve(control_ensemble_prob, Tsit5(), saveat=(length(acc)-1):(length(acc)-1+preddays), EnsembleThreads(), trajectories=1000)
    return controlsim, outputdate, controlρ, controlσ, controlh, controlSS
end

function senarioplot!(sol, controlsol, preddays, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
    sim, simdate = sol
    accmean = timeseries_steps_mean(sim)[10, :]
    dailymean = computedaily!(accmean)
    accup = timeseries_steps_quantile(sim, 0.95)[10, :]
    dailyup = computedaily!(accup)
    acclow = timeseries_steps_quantile(sim, 0.05)[10, :]
    dailylow = computedaily!(acclow)
    ## Accumulated cases
    scatter(datadate, acc[datatspan], markercolor=markercolor, label="Observed acculated cases", legend_background_color=nothing, title=country)
    plot!(simdate, accmean, lw=3, linecolor=linecolor, color=color, ribbon=(acclow, accup), fillalpha=0.3, label="Predicted accumulated cases", legend_background_color=nothing, title=country)
    for case in 1:length(controlsol)
        controlsim, controldate, controlρ, controlσ, controlh, controlSS = controlsol[case]
        controlaccmean = timeseries_steps_mean(controlsim)[10, :]
        #controldailymean = computedailycontrol!(controlaccmean)
        controlaccup = timeseries_steps_quantile(controlsim, 0.95)[10, :]
        #controldailyup = computedailycontrol!(controlaccup)
        controlacclow = timeseries_steps_quantile(controlsim, 0.05)[10, :]
        #controldailylow = computedailycontrol!(controlacclow)
        plot!(controldate, controlaccmean, lw=3, ribbon=(controlacclow, controlaccup), fillalpha=0.3, label="Forecasting accumulated cases $controlρ ρ $controlσ σ", legend_background_color=nothing, title=country)
    end
    #savefig("./output/senocontrolacc$i.png")
    savefig("./output/senocontrolaccsigma$i.png")
    ## daily
    scatter(datadate, cases[datatspan], markercolor=markercolor, label="Observed daily cases", legend_background_color=nothing, title=country)
    plot!(simdate, dailymean, lw=3, linecolor=linecolor, color=color, ribbon=(dailylow, dailyup), fillalpha=0.3, label="Predicted daily cases", legend_background_color=nothing, title=country)
    for case in 1:length(controlsol)
        controlsim, controldate, controlρ, controlσ, controlh, controlSS = controlsol[case]
        controlaccmean = timeseries_steps_mean(controlsim)[10, :]
        controldailymean = computedailycontrol!(controlaccmean)
        controlaccup = timeseries_steps_quantile(controlsim, 0.95)[10, :]
        controldailyup = computedailycontrol!(controlaccup)
        controlacclow = timeseries_steps_quantile(controlsim, 0.05)[10, :]
        controldailylow = computedailycontrol!(controlacclow)
        plot!(controldate[2:end], controldailymean, lw=3, ribbon=(controldailylow, controldailyup), fillalpha=0.3, label="Forecasting daily cases,$controlρ ρ $controlσ σ", legend_background_color=nothing, title=country)
    end
    #savefig("./output/senocontroldaily$i.png") # rho
    savefig("./output/senocontroldailysigma$i.png") # sigma
end

##
i = 1
country = countryarray[i]
url = "./data/timeseries-country-confirmed.csv";
color = :cornflowerblue
linecolor = :darkturquoise
markercolor = :darksalmon
preddays = 60
controlρ = [0.5,0.2,0.1,0.01]
controlσ = 1.0
controlh = 1.0
controlSS = 1.0
sol = solinference!(i, url, country)
sim = solinference!(i, url, country)[1]
controlsol1 = solcontrolinference!(sim, i, url, country, preddays, controlρ[1], controlσ, controlh, controlSS)
controlsol2 = solcontrolinference!(sim, i, url, country, preddays, controlρ[2], controlσ, controlh, controlSS)
controlsol3 = solcontrolinference!(sim, i, url, country, preddays, controlρ[3], controlσ, controlh, controlSS)
controlsol4 = solcontrolinference!(sim, i, url, country, preddays, controlρ[4], controlσ, controlh, controlSS)
#controlsol5 = solcontrolinference!(sim, i, url, country, preddays, controlρ[5], controlσ, controlh, controlSS)
#controlsol6 = solcontrolinference!(sim, i, url, country, preddays, controlρ[6], controlσ, controlh, controlSS)
controlsol = (controlsol1, controlsol2, controlsol3, controlsol4)#, controlsol5)#, controlsol6)
senarioplot!(sol, controlsol, preddays, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)



##

##
i = 1
country = countryarray[i]
url = "./data/timeseries-country-confirmed.csv";
color = :cornflowerblue
linecolor = :darkturquoise
markercolor = :darksalmon
preddays = 60
controlρ = 1.0
controlσ = [1.0,2.0,4.0,6.0]
controlh = 1.0
controlSS = 1.0
sol = solinference!(i, url, country)
sim = solinference!(i, url, country)[1]
controlsol1 = solcontrolinference!(sim, i, url, country, preddays, controlρ, controlσ[1], controlh, controlSS)
controlsol2 = solcontrolinference!(sim, i, url, country, preddays, controlρ, controlσ[2], controlh, controlSS)
controlsol3 = solcontrolinference!(sim, i, url, country, preddays, controlρ, controlσ[3], controlh, controlSS)
controlsol4 = solcontrolinference!(sim, i, url, country, preddays, controlρ, controlσ[4], controlh, controlSS)
#controlsol5 = solcontrolinference!(sim, i, url, country, preddays, controlρ[5], controlσ, controlh, controlSS)
#controlsol6 = solcontrolinference!(sim, i, url, country, preddays, controlρ[6], controlσ, controlh, controlSS)
controlsol = (controlsol1, controlsol2, controlsol3, controlsol4)#, controlsol5)#, controlsol6)
senarioplot!(sol, controlsol, preddays, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)






##
for i in 1:length(countryarray)
    if i>1
    country = countryarray[i]
    url = "./data/timeseries-country-confirmed.csv";
    color = :cornflowerblue
    linecolor = :darkturquoise
    markercolor = :darksalmon
    preddays = 60
    controlρ = [0.5,0.2,0.1,0.01]
    controlσ = 1.0
    controlh = 1.0
    controlSS = 1.0
    sol = solopt!(i, url, country)
    sim = sol[1]
    controlsol1 = solcontrolopt!(sim, i, url, country, preddays, controlρ[1], controlσ, controlh, controlSS)
    controlsol2 = solcontrolopt!(sim, i, url, country, preddays, controlρ[2], controlσ, controlh, controlSS)
    controlsol3 = solcontrolopt!(sim, i, url, country, preddays, controlρ[3], controlσ, controlh, controlSS)
    controlsol4 = solcontrolopt!(sim, i, url, country, preddays, controlρ[4], controlσ, controlh, controlSS)
    #controlsol5 = solcontrolopt!(sim, i, url, country, preddays, controlρ[5], controlσ, controlh, controlSS)
    #controlsol6 = solcontrolopt!(sim, i, url, country, preddays, controlρ[6], controlσ, controlh, controlSS)
    controlsol = (controlsol1, controlsol2, controlsol3, controlsol4)#, controlsol5, controlsol6)
    senarioplot!(sol, controlsol, preddays, color, linecolor, markercolor, datadate, datatspan, i, country, acc, cases)
    end
end