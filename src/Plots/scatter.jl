struct Scatter{T<:NamedTuple} <: AbstractPlot
    nt::T
end

Scatter(x) = Scatter((_=x,))

function plot!(ax, scatter::Scatter; kwargs...)
    @unpack nt = scatter
    alpha = length(nt) > 0.75 ? 0.5 : 1.0
    for (x, label) in zip(values(nt), keys(nt))
        ax.scatter([x[i,:] for i in 1:size(x, 1)]..., marker=".", alpha=alpha, label=label)
    end
    autoset_lims!(ax, first(values(nt)))
    length(nt) > 1 && ax.legend(fancybox=true, framealpha=0.5)
    return ax
end
