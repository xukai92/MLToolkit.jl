import ArgParse: parse_args

function parse_args(args_str::AbstractString, settings; as_symbols::Bool=false)
    parse_args(split(replace(args_str, r"\s+" => " "), " "), settings; as_symbols=as_symbols)
end

function flatten_dict(dict::Dict{T,Any};
                      equal_sym="=",
                      delimiter="-",
                      exclude::Vector{T}=T[],
                      include::Vector{T}=collect(keys(dict))) where {T<:Union{String,Symbol}}
    @assert issubset(Set(exclude), keys(dict)) "Keyword `exclude` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(exclude), keys(dict)))"
    @assert issubset(Set(include), keys(dict)) "Keyword `include` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(include), keys(dict)))"
    return join(["$k$equal_sym$v" for (k,v) in filter(t -> (t[1] in include) && !(t[1] in exclude), dict)], delimiter)
end

"""
    @jupyter expr

Execute `expr` if IN Jupyter.
"""
macro jupyter(expr)
    return :(isdefined(Main, :IJulia) && Main.IJulia.inited && $expr)
end

"""
    @script expr

Execute `expr` if NOT IN Jupyter.
"""
macro script(expr)
    return :(!(isdefined(Main, :IJulia) && Main.IJulia.inited) && $expr)
end

"""
    checknumerical(vcheck, vmonitor...; vcheckname=nothing, vmonitornames=nothing)

Check if each entry in `vcheck` is `NaN` or `Inf`.
If so, report the index and the value of that index in all variables.
"""
function checknumerical(vcheck, vmonitor...; vcheckname=nothing, vmonitornames=nothing)
    for i in eachindex(vmonitor)
        @assert size(vcheck) == size(vmonitor[i]) "All variables in `vmonitor` should have the same size as `vcheck`; vmonitor[$i] doesn't."
    end
    for i in eachindex(vcheck)
        check = vcheck[i]
        if isnan(check) || isinf(check)
            if vcheckname != nothing
                check = Dict("$vcheckname[$i]" => check)
            end
            if length(vmonitor) == 0
                @info "Numerical error found in index $i" check
            else
                monitor = map(v -> v[i], vmonitor)
                if vmonitornames != nothing
                    monitor = Dict(zip(map(mn -> "$mn[$i]", vmonitornames), monitor))
                end
                @info "Numerical error found in index $i" check monitor
            end
        end
    end
end

"""
    @checknumerical vcheck vmonitor1 vmonitor2 ...

Helper macro to call `checknumerical` with variable names extracted.
"""
macro checknumerical(vcheck, vmonitor...)
    vcheckname = String(vcheck)
    vmonitornames = map(m -> String(m), vmonitor)
    return quote
        checknumerical($vcheck, $(vmonitor...); vcheckname=$vcheckname, vmonitornames=$vmonitornames)
    end
end