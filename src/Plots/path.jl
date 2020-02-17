struct TwoDimPath <: AbstractPlot
    xs
    ys
end

function plot!(
    ax, p::TwoDimPath, linestyle="-"; 
    first=nothing, last=nothing, title=nothing, kwargs...
)
    ax.plot(p.xs, p.ys, linestyle, kwargs...)
    !isnothing(first) && ax.scatter(
        p.xs[1], p.ys[1]; facecolor="none", marker=first.marker, color=first.color
    )
    !isnothing(last) && ax.scatter(
        p.xs[end], p.ys[end]; facecolor="none", marker=last.marker, color=last.color
    )
    !isnothing(title) && ax.set_title(title)
end

