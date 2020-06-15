### MultipleLines

using Statistics: mean, std, var

struct ErrorBarLines
    x
    ys
end

@recipe function f(lines::ErrorBarLines; nstd=1, std_warn=nothing)
    @unpack x, ys = lines
    
    y = mean(ys)
    y_s = std(ys)
    dy = nstd * y_s
    
    # Add a series for an error band
    @series begin
        seriestype := :path
        # Ignore series in legend and color cycling
        primary := false
        linecolor := nothing
        fillcolor := :lightgray
        fillalpha := 0.5
        fillrange := y .- dy
        # ensure no markers are shown for the error band
        markershape := :none
        # Return series data
        x, y + dy
    end
    
    # Highlight `s` larger than `σ_warn` via :x as markers
    # NOTE: this is NOT properly supported by PGFPlotsX
    if !isnothing(std_warn)
        s = get(plotattributes, :markershape, :none)
        markershape := ifelse.(y_s .> std_warn, :xcross, s)
    end

    x, y
end

### TwoYAxesLines

using Plots: mm
import Plots: plot

struct TwoYAxesLines
    x
    y1
    y2
end

function plot(
    lines::TwoYAxesLines;
    color1=1,
    color2=2,
    xlabel=nothing,
    ylabel1=nothing,
    ylabel2=nothing,
    kwargs...
)
    @unpack x, y1, y2 = lines
    right_margin = isnothing(ylabel2) ? :match : 13mm
    p = plot(x, y1; color=color1, xlabel=xlabel, ylabel=ylabel1, right_margin=right_margin, kwargs...)
    plot!(twinx(p), x, y2; color=color2, ylabel=ylabel2, kwargs...)
    return p
end
