using ..MLToolkit.Plots

function vis(dataset::AbstractDataset, args...; kwargs...)
    fig, ax = plt.subplots(figsize=(5, 5))
    vis!(ax, dataset, args...; kwargs...)
    return fig
end

function vis(dataset::AbstractDataset{3}, args...; kwargs...)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    vis!(ax, dataset, args...; kwargs...)
    return fig
end

vis!(ax, d::AbstractDataset, x) = vis!(ax, d, (x=x,))

function vis!(ax, ::Union{AbstractDataset{2}, AbstractDataset{3}}, nt::NamedTuple)
    alpha = length(nt) > 0.75 ? 0.5 : 1.0
    for (x, label) in zip(values(nt), keys(nt))
        ax.scatter([x[i,:] for i in 1:size(x, 1)]..., marker=".", alpha=alpha, label=label)
    end
    autoset_lims!(ax, first(values(nt)))
    length(nt) > 1 && ax.legend(fancybox=true, framealpha=0.5)
end

function vis!(ax, d::ImageDataset, nt::NamedTuple{T1, <:NTuple{N, T2}}) where {T1, N, T2}
    plot!(ax, ImageGrid(invlink(d, cat(values(nt)...; dims=ndims(T2)))))
end
