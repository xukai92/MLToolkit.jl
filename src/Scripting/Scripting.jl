module Scripting

namedtuple2dict(t::NamedTuple) = Dict(pairs(t)...)
dict2namedtuple(d::Dict) = NamedTuple{tuple(keys(d)...),typeof(tuple(values(d)...))}(tuple(values(d)...))

function Base.reduce(op::Function, ts::Union{AbstractVector{T}, Tuple{Vararg{T}}}) where {T<:NamedTuple}
    return T(tuple(map(op, zip(values.(ts)...))...))
end

"""
    sweepcmd(template, sweeps)

Generate a list of commands given a command template and sweep mappings.

### Usage

```julia
sweepcmd("sleep @Ts", "@T" => [1, 2, 3])
sweepcmd("sleep @Ts", :T => [1, 2, 3])
```
"""
function sweepcmd(template, sweeps::Pair...)
    n_sweeps = length(sweeps)
    names = map(s -> s.first isa Symbol ? "@$(s.first)" : s.first, sweeps)
    vlists = map(s -> s.second, sweeps)
    cmds = Cmd[]
    for values in Base.Iterators.product(vlists...)
        cmdstr = template
        for i = 1:n_sweeps
            cmdstr = replace(cmdstr, names[i] => string(values[i]))
        end
        cmd = Cmd(map(String, split(cmdstr, " ")))
        push!(cmds, cmd)
    end
    return cmds
end

"""
    sweepcmd(template, sweeps)

Run a list of commands given a command template and sweep mappings.

### Usage

```julia
sweeprun("sleep @Ts", "@T" => [1, 2, 3])
sweeprun("sleep @Ts", :T => [1, 2, 3])
```
"""
function sweeprun(template, sweeps::Pair...; maxasync=0)
    cmds = sweepcmd(template, sweeps...)
    for cmds_partitioned in Iterators.partition(cmds, maxasync > 0 ? maxasync : length(cmds))
        @sync begin
            for cmd in cmds_partitioned
                @async run(cmd)
            end
        end
    end
end

export namedtuple2dict, dict2namedtuple, sweepcmd, sweeprun

include("args.jl")
export argstring, argstring_flat, DATETIME_FMT, find_latestdir, parse_toml, parse_argstr

include("check.jl")
export isjupyter, istb, @jupyter, @script, @tb, checknumerics, @checknumerics

end # module
