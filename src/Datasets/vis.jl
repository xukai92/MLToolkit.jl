using ..MLToolkit.PlotsX
using ..MLToolkit.PlotsX: plot, plot!

function vis(dataset::AbstractDataset, args...; kwargs...)
    p = plot()
    vis!(p, dataset, args...; kwargs...)
    return p
end

vis!(p, d::AbstractDataset, x) = vis!(p, d, (x=x,))

function vis!(p, ::Union{AbstractDataset{2}, AbstractDataset{3}}, nt::NamedTuple)
    plot!(p, LabelledScatter(nt))
end

function vis!(p, d::ImageDataset, nt::NamedTuple{T1, <:NTuple{N, T2}}) where {T1, N, T2<:AbstractArray}
    plot!(p, ImageGrid(invlink(d, cat(values(nt)...; dims=ndims(T2)))))
end
