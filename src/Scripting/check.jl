isjupyter() = isdefined(Main, :IJulia) && Main.IJulia.inited

"""
    @jupyter expr

Execute `expr` if IN Jupyter.
"""
macro jupyter(expr)
    if isjupyter()
        return esc(expr)
    else
        return nothing
    end
end

"""
    @script expr

Execute `expr` if NOT IN Jupyter.
"""
macro script(expr)
    if !isjupyter()
        return esc(expr)
    else
        return nothing
    end
end

"""
    checknumerics(vcheck, vmonitor...; vcheckname=nothing, vmonitornames=nothing)

Check if each entry in `vcheck` is `NaN` or `Inf`.
If so, report the index and the value of that index in all variables.
"""
function checknumerics(vcheck, vmonitor...; vcheckname=nothing, vmonitornames=nothing)
    for i in eachindex(vmonitor)
        @assert size(vcheck) == size(vmonitor[i]) "All variables in `vmonitor` should have the same size as `vcheck`; vmonitor[$i] doesn't."
    end
    for i in eachindex(vcheck)
        check = vcheck[i]
        if isnan(check) || isinf(check)
            ci = CartesianIndices(size(vcheck))[i]
            Istr = replace(string(ci.I), r"(\s)|(\()|(\))" => "")
            if vcheckname != nothing
                check = ("($vcheckname[$Istr], $check)")
            end
            if length(vmonitor) == 0
                @info "Numerical error found in index [$Istr]" check
            else
                monitor = map(v -> v[i], vmonitor)
                if vmonitornames != nothing
                    monitor = collect(zip(map(mn -> "$mn[$Istr]", vmonitornames), monitor))
                    monitor = map(t -> "($(t[1]), $(t[2]))", monitor)
                    monitor = join(monitor, " ")
                end
                @info "Numerical error found in index [$Istr]" check monitor
            end
        end
    end
end

"""
    @checknumerics scheck smonitor1 smonitor2 ...

Helper macro to call `checknumerics` with variable names extracted and passed.
"""
macro checknumerics(scheck, smonitor...)
    local vcheck = esc(scheck)
    local vmonitor = map(esc, smonitor)
    vcheckname = string(scheck)
    vmonitornames = map(m -> string(m), smonitor)
    return quote
        checknumerics($vcheck, $(vmonitor...); vcheckname=$vcheckname, vmonitornames=$vmonitornames)
    end
end

### Logging

import Logging

istb() = Logging.current_logger() isa TensorBoardLogger.TBLogger

"""
    @tb expr

Execute `expr` if the current logger is TBLogger.
"""
macro tb(expr)
    if istb()
        return esc(expr)
    else
        return nothing
    end
end
