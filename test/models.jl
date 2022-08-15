using Monkeypox
using Plots
url = "./data/timeseries-country-confirmed.csv"
country = "United States"
data_on, acc, cases, datatspan, datadate = datasource!(url, country)
##
## initial values and parameters
#N = 38010000.0
N = 329500000.0 # population
θ = [0.3, 0.3, 0.7, 0.01]# ρ,σ,h,α
pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
## run monkeypoxpair
prob = monkeypoxprob!(N, θ, acc, pknown)

src = simulate!(prob, N, θ, datatspan, pknown)
plot(src[5,:])

##
## initial values and parameters
N = 38010000.0
#N = 329500000.0 # population
θ = [0.3, 0.3, 0.2, 0.05, 0.7, 0.01]# ρ,σ,h,α

pknown = [0.0, 0.0, 1 / 30, 1.0] # B,μ,δ,ϕ
## run monkeypoxpair
prob = controlmonkeypoxprob!(N, θ, acc, pknown)

src = controlsimulate!(prob, N, θ, datatspan, pknown)
plot(src[2,:])