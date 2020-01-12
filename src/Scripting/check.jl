"""
    @jupyter expr default=nothing

Execute `expr` if it's currently in Jupyter.
"""
macro jupyter(expr, default=nothing)
    return esc(
        quote
            if isdefined(Main, :IJulia) && Main.IJulia.inited
                return $expr
            else
                return $default
            end
        end
    )
end

"""
    @script expr default=nothing

Execute `expr` if it's currently not in Jupyter.
"""
macro script(expr, default=nothing)
    return esc(
        quote
            if !(isdefined(Main, :IJulia) && Main.IJulia.inited)
                return $expr
            else
                return $default
            end
        end
    )
end

"""
    @tb expr default=nothing

Execute `expr` if the current logger is a `TBLogger`.
"""
macro tb(expr, default=nothing)
    return esc(
        quote
            !isdefined(Main, :current_logger) && import Logging
            !isdefined(Main, :TBLogger) && import TensorBoardLogger
            if Logging.current_logger() isa TensorBoardLogger.TBLogger
                return $expr
            else
                return $default
            end
        end
    )
end

"""
    @wb expr default=nothing

Execute `expr` if the current logger is a `WBLogger`.
"""
macro wb(expr, default=nothing)
    return esc(
        quote
            !isdefined(Main, :current_logger) && import Logging
            !isdefined(Main, :WBLogger) && import WeightsAndBiasLogger
            if Logging.current_logger() isa WeightsAndBiasLogger.WBLogger
                return $expr
            else
                return $default
            end
        end
    )
end

function checknumerics_strs(vcheck, vmonitor...; vcheckname=nothing, vmonitornames=["monitor$i" for i in 1:length(vmonitor)])
    for i in eachindex(vmonitor)
        @assert size(vcheck) == size(vmonitor[i]) "All variables in `vmonitor` should have the same size as `vcheck`; vmonitor[$i] doesn't."
    end
    # Get indics for NaN and Inf
    idcs = filter(eachindex(vcheck)) do i
        isnan(vcheck[i]) || isinf(vcheck[i])
    end
    # Convert to Cartesian
    c = CartesianIndices(size(vcheck))
    warnstr_list = []
    for i in idcs
        Istr = string(c[i].I)
        Istr_flat = replace(Istr, r"(\s)|(\()|(\))" => "")  # "(1, 2)" => "1,2"
        # Make checkstr
        if isnothing(vcheckname)
            warnstr = "Index $Istr is $(vcheck[i])"
        else
            warnstr = "$vcheckname[$Istr_flat] = $(vcheck[i])"
        end
        if length(vmonitor) > 0
            # Make monitorstr
            monitorstr = map(v -> v[i], vmonitor)
            monitorstr = collect(zip(map(mn -> "$mn[$Istr_flat]", vmonitornames), monitorstr))
            monitorstr = map(t -> "$(t[1]) = $(t[2])", monitorstr)
            monitorstr = join(monitorstr, "\n  ")
            warnstr = "$warnstr\n  $monitorstr"
        end
        push!(warnstr_list, warnstr)
    end
    return warnstr_list
end

"""
    @checknumerics scheck smonitor...

Check if each entry in `scheck` is `NaN` or `Inf`.
If so, report the index and the value of that index in all variables.

Usage:
```julia
vc = [1 Inf 3; NaN 5 6]
vm1 = [1 2 3; 3 4 5]
vm2 = [11 22 33; 44 55 66]
@checknumerics vc vm1 vm2
```
"""
macro checknumerics(scheck, smonitor...)
    local vcheck = esc(scheck)
    local vmonitor = esc.(smonitor)
    vcheckname = string(scheck)
    vmonitornames = string.(smonitor)
    return quote
        for warnstr in checknumerics_strs($vcheck, $(vmonitor...); vcheckname=$vcheckname, vmonitornames=$vmonitornames)
            @warn warnstr
        end
    end
end
