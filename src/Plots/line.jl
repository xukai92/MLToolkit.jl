"""
A plot of two lines with a shared y-axis.
"""
@with_kw struct TwoYAxesLines <: AbstractPlot
    x
    y1
    y2
    fmt="-"
    colour1="red"
    colour2="blue"
    xlabel=nothing
    ylabel1=nothing
    ylabel2=nothing
end

function plot(lines::TwoYAxesLines)
    @unpack x, y1, y2, fmt, colour1, colour2, xlabel, ylabel1, ylabel2 = lines
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, fmt, c=colour1)
    xlabel == nothing || ax1."set_xlabel"(xlabel)
    ylabel1 == nothing || ax1."set_ylabel"(ylabel1, color=colour1)
    ax1.tick_params(axis="y", labelcolor=colour1)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, fmt, c=colour2)
    ylabel2 == nothing || ax2."set_ylabel"(ylabel2, color=colour2)
    ax2.tick_params(axis="y", labelcolor=colour2)
    fig.tight_layout()
    return fig
end

function get_tikz_code(lines::TwoYAxesLines)
    fig = plot(lines)
    tikz_code = tikzplotlib.get_tikz_code()
    tikz_code = replace(tikz_code, "tick pos=both" => "tick pos=left";  count=1)
    tikz_code = replace(tikz_code, "tick pos=both" => "tick pos=right"; count=1)
    tikz_code = replace(tikz_code, "axis y line=right" => "yticklabel pos=right, ylabel style={rotate=180}, xtick=\\empty"; count=1)
    return tikz_code
end