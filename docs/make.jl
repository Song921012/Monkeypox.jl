using Monkeypox
using Documenter

DocMeta.setdocmeta!(Monkeypox, :DocTestSetup, :(using Monkeypox); recursive=true)

makedocs(;
    modules=[Monkeypox],
    authors="Pengfei Song",
    repo="https://github.com/Song921012/Monkeypox.jl/blob/{commit}{path}#{line}",
    sitename="Monkeypox.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Song921012.github.io/Monkeypox.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Song921012/Monkeypox.jl",
    devbranch="master",
)
