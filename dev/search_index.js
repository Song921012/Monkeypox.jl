var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Monkeypox","category":"page"},{"location":"#Monkeypox","page":"Home","title":"Monkeypox","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Monkeypox.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Monkeypox]","category":"page"},{"location":"#Monkeypox.datasource!-Tuple{Any, Any}","page":"Home","title":"Monkeypox.datasource!","text":"datasource!(url, country)\n\ndata processing\n\n\n\n\n\n","category":"method"},{"location":"#Monkeypox.monkeypoxopt!-NTuple{9, Any}","page":"Home","title":"Monkeypox.monkeypoxopt!","text":"monkeypoxopt!(N::Real, θ, acc, cases, datatspan, pknown, lb, ub, alg)\n\nParameter Estimation\n\nParameters: (population, estimated parameters, acc,cases,datatspan,pknown, lb,ub,alg)\n\n\n\n\n\n","category":"method"},{"location":"#Monkeypox.monkeypoxpair!-NTuple{4, Any}","page":"Home","title":"Monkeypox.monkeypoxpair!","text":"monkeypoxpair!(du,u,p,t)\n\nDefine classic Monkeypox compartment model. \n\nParameters: (population, infection rate, recovery rate)\n\nbeginalignfracdS(t)dt = B + left( mu + sigma right) left( 2 mathrmSSleft( t right) + mathrmSIleft( t right) + mathrmSRleft( t right) right) - left( mu + rho right) Sleft( t righ  1481 bytes  sigma + 2 mu right) mathrmRRleft( t right) fracdH(t)dt = fracrho left( 1 - h right) Ileft( t right) Sleft( t right)N + frac2 h rho Ileft( t right) Sleft( t right)Nendalign\n\n\n\n\n\n","category":"method"},{"location":"#Monkeypox.monkeypoxprob!-NTuple{4, Any}","page":"Home","title":"Monkeypox.monkeypoxprob!","text":"monkeypoxprob!(N, θ, acc, pknown)\n\ngenerate monkeypox ode problem\n\n\n\n\n\n","category":"method"},{"location":"#Monkeypox.simulate!-Tuple{SciMLBase.ODEProblem, Any, Any, Any, Any}","page":"Home","title":"Monkeypox.simulate!","text":"simulate!(prob::ODEProblem,N, θ, datatspan,pknown)\n\nsolve monkeypox model\n\n\n\n\n\n","category":"method"}]
}
