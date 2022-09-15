using PackageCompiler
@time create_sysimage([:DifferentialEquations,:Turing,:Plots,:DataFrames,:Optimization,:SciMLSensitivity,:Monkeypox], sysimage_path="JuliaSysimage.dll", precompile_execution_file="./test/inference.jl")


using Pkg, Dates
 Pkg.gc(collect_delay = Day(0))