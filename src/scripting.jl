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