using ..MLToolkit.PlotRecipes

function vis(dataset::AbstractDataset, args...; kwargs...)
    fig, ax = figure(figsize=(5, 5))
    vis!(ax, dataset, args...; kwargs...)
    return fig
end

function vis(dataset::AbstractDataset{3}, args...; kwargs...)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1; projection="3d")
    vis!(ax, dataset, args...; kwargs...)
    return fig
end

vis!(p, d::AbstractDataset, x) = vis!(p, d, (x=x,))

function vis!(p, ::Union{AbstractDataset{2}, AbstractDataset{3}}, nt::NamedTuple)
    plot!(p, LabelledScatter(nt))
end

function vis!(p, d::ImageDataset, nt::NamedTuple{T1, <:NTuple{N, T2}}) where {T1, N, T2<:AbstractArray}
    plot!(p, ImageGrid(invlink(d, cat(values(nt)...; dims=ndims(T2)))))
end
