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

macro jupyter(expr)
    return :(isdefined(Main, :IJulia) && Main.IJulia.inited && $expr)
end

macro script(expr)
    return :(!(isdefined(Main, :IJulia) && Main.IJulia.inited) && $expr)
end