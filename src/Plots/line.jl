"""
A plot of two lines with a shared y-axis.
"""
struct TwoYAxesLines <: AbstractPlot
    x
    y1
    y2
end

function plot(
    lines::TwoYAxesLines,
    fmt="-";
    colour1="red",
    colour2="blue",
    xlabel=nothing,
    ylabel1=nothing,
    ylabel2=nothing,
    kwargs...
)
    @unpack x, y1, y2 = lines
    fig, ax1 = figure()
    ax1.plot(x, y1, fmt; c=colour1, kwargs...)
    xlabel == nothing || ax1.set_xlabel(xlabel)
    ylabel1 == nothing || ax1.set_ylabel(ylabel1; color=colour1)
    ax1.tick_params(axis="y"; labelcolor=colour1)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, fmt, c=colour2, kwargs...)
    ylabel2 == nothing || ax2.set_ylabel(ylabel2; color=colour2, rotation="-90")
    ax2.tick_params(axis="y"; labelcolor=colour2)
    fig.tight_layout()
    return fig
end

function get_tikz_code(fig::PyPlot.Figure, lines::TwoYAxesLines)
    tikz_code = tikzplotlib.get_tikz_code(fig)
    tikz_code = replace(tikz_code, "tick pos=both" => "tick pos=left";  count=1)
    tikz_code = replace(tikz_code, "tick pos=both" => "tick pos=right"; count=1)
    tikz_code = replace(tikz_code, "axis y line=right" => "yticklabel pos=right, xtick=\\empty"; count=1)
    return tikz_code
end

"""
A plot of 1D function.
"""
struct OneDimFunction <: AbstractPlot
    f
end

function plot!(
    ax,
    odf::OneDimFunction,
    startstop;
    nbins=1_000,
    kwargs...
)
    f = odf.f
    x = range(startstop...; length=nbins)
    y = f.(x)
    ax.plot(x, y; kwargs...)
    return ax
end

using Statistics: mean, std

"""
A plot of lines with shaded error bar.
"""
struct LinesWithErrorBar <: AbstractPlot
    x
    y_mean
    y_std
end

LinesWithErrorBar(x, ys) = LinesWithErrorBar(x, mean(ys), std(ys))

function plot!(
    ax,
    linesbar::LinesWithErrorBar;
    nstd=1,
    alpha=0.5,
    kwargs...
)
    @unpack x, y_mean, y_std = linesbar
    y = y_mean
    dy = nstd * y_std
    ax.plot(x, y; kwargs...)
    ax.fill_between(x, y - dy, y + dy; alpha=alpha)
    return ax
end
