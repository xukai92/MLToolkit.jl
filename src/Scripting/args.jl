# argstr        ::  String      :   "--x 1 --y 2"
# argstr_flat   ::  String      :   "x=1-y=2"
# argdict       ::  Dict        :   Dict(:x => 1, :y => 2)
# args          ::  NamedTuple  :   (x=1, y=2)

function argstring(
    dict::Dict{T,<:Any},
    prefix,
    eqsym,
    delimiter;
    exclude::Vector{T}=T[],
    include::Vector{T}=collect(keys(dict))
) where {T<:Union{String,Symbol}}
    @assert issubset(Set(exclude), keys(dict)) "Keyword `exclude` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(exclude), keys(dict)))"
    @assert issubset(Set(include), keys(dict)) "Keyword `include` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(include), keys(dict)))"
    return join([v == "" ? "$prefix$k" : "$prefix$k$eqsym$v" for (k, v) in filter(t -> (t[1] in include) && !(t[1] in exclude), dict)], delimiter)
end

argstring(argdict::Dict; kwargs...) = argstring(argdict, "--", " ", " "; kwargs...)
argstring_flat(argdict::Dict; eqsym="=", delimiter="-", kwargs...) = argstring(argdict, "", eqsym, delimiter; kwargs...)

import Dates

const DATETIME_FMT = "ddmmyyyy-H-M-S"

function find_latestdir(parentdir)
    dates = []
    for dirstr in readdir(parentdir)
        try
            push!(dates, Dates.DateTime(dirstr, DATETIME_FMT))
        catch
            nothing
        end
    end
    return Dates.format(maximum(dates), DATETIME_FMT)
end

using Pkg.TOML

function parse_toml(tomlpath::String, tableinfo::Tuple{Vararg{Pair}})
    # Load TOML file into nested dictionary
    toml = TOML.parsefile(tomlpath)
    # Get the "common" section
    argdict = toml["common"]
    local table = toml
    for (k, v) in tableinfo
        table = table[v]
        argdict = merge(argdict, filter(p -> !(p.second isa Dict), table))
        argdict[string(k)] = v
    end
    # Check if there are sub-tables left to read
    @assert length(filter(p -> p.second isa Dict, table)) == 0 "There sub-table left in the current nesting table."
    # Convert keys to Symbol
    return Dict(Symbol(p.first) => p.second for p in argdict)
end

import ArgParse

function parse_argstr(argstr::AbstractString, settings; as_symbols::Bool=false)
    return ArgParse.parse_args(split(replace(argstr, r"\s+" => " "), " "), settings; as_symbols=as_symbols)
end
