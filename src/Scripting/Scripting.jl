module Scripting

using DrWatson

Base.:(+)(nt1::T, nt2::T) where {T<:NamedTuple} = T(tuple((values(nt1) .+ values(nt2))...))

function Base.union(nt1::NamedTuple{S1, T1}, nt2::NamedTuple{S2, T2}) where {S1, T1, S2, T2}
    return NamedTuple{(S1..., S2...), Tuple{T1.types..., T2.types...}}((nt1..., nt2...))
end

# Conventions
#   args        ::  NamedTuple  :   (x=1, y=2)
#   argdict     ::  Dict        :   Dict(:x => 1, :y => 2)
#   savename    ::  String      :   "x=1-y=2"
#   cmdstr      ::  String      :   "--x 1 --y 2"

function overwrite(args, override)
    argdict = ntuple2dict(args)
    for k in keys(override)
        if k in keys(argdict)
            v = override[k]
            if argdict[k] == v
                @warn "The values for key :$k are the same: $v; overwriting is skipped."
            else
                @info "Overriding :$k: $(args[k]) => $v"
            end
            argdict[k] = v
        else
            @warn "Key $k doesn't exist in `args`; overwriting is skipped."
        end
    end
    return dict2ntuple(argdict)
end

cmdstr(args::NamedTuple) = cmdstr(ntuple2dict(args))
function cmdstr(argdict::Dict)
    return replace(savename(Dict("--$k" => v for (k, v) in argdict); connector=" "), "=" => " ")
end

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

export overwrite, cmdstr, DATETIME_FMT, find_latestdir, parse_toml, parse_argstr

include("check.jl")
export @jupyter, @script, @tb, @wb, @checknumerics

end # module
