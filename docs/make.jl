MLT_PATH = joinpath(splitpath(@__DIR__)[1:end-1]...)
push!(LOAD_PATH, MLT_PATH)

using Documenter, MLToolkit

makedocs(;
    sitename = "MLToolkit Documentation",
    pages = [
        "Home" => "index.md",
        "Modules" => [
            "plots.md",
        ]
    ]
)

deploydocs(
    repo = "https://github.com/xukai92/MLToolkit.jl.git",
)
