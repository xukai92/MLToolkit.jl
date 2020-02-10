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
    plot!(ax, Scatter(nt))
end

function vis!(ax, d::ImageDataset, nt::NamedTuple{T1, <:NTuple{N, T2}}) where {T1, N, T2}
    plot!(ax, ImageGrid(invlink(d, cat(values(nt)...; dims=ndims(T2)))))
end
