import ArgParse: parse_args

const DATETIME_FMT = "ddmmyyyy-H-M-S"

function find_latest_dir(targetdir)
    date_list = []
    for dirstr in readdir(targetdir)
        try
            push!(date_list, Dates.DateTime(dirstr, DATETIME_FMT))
        catch
            nothing
        end
    end
    return Dates.format(maximum(date_list), DATETIME_FMT)
end

function parse_args(args_str::AbstractString, settings; as_symbols::Bool=false)
    parse_args(split(replace(args_str, r"\s+" => " "), " "), settings; as_symbols=as_symbols)
end

function flatten_dict(dict::Dict{T,<:Any};
                      equal_sym="=",
                      delimiter="-",
                      exclude::Vector{T}=T[],
                      include::Vector{T}=collect(keys(dict))) where {T<:Union{String,Symbol}}
    @assert issubset(Set(exclude), keys(dict)) "Keyword `exclude` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(exclude), keys(dict)))"
    @assert issubset(Set(include), keys(dict)) "Keyword `include` must be a subset of `keys(dict)`; set diff: $(setdiff(Set(include), keys(dict)))"
    return join(["$k$equal_sym$v" for (k,v) in filter(t -> (t[1] in include) && !(t[1] in exclude), dict)], delimiter)
end

function dict2namedtuple(d)
    return NamedTuple{tuple(keys(d)...),typeof(tuple(values(d)...))}(tuple(values(d)...))
end

function merge_namedtuples(op, t1, t2)
    return NamedTuple{tuple(keys(t1)...),typeof(tuple(values(t1)...))}(tuple(map(op, zip(values(t1), values(t2)))...))
end

args_dict2str(args_dict) = join([v == "" ? "--$k" : "--$k $v" for (k, v) in args_dict], " ")

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

istb() = Logging.current_logger() isa TensorBoardLogger.TBLogger

"""
    @tb expr

Execute `expr` if the current logger is TBLogger.
"""
macro tb(expr)
    return esc(
        quote
            if istb()
                $expr
            end
        end
    )
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

"""
    sweepcmd(cmd_template, sweeps)

Generate a list of commands given a command template and sweep mappings.

Example:

```julia
sweepcmd("sleep @Ts", "@T" => [1, 2, 3])
sweepcmd("sleep @Ts", :T => [1, 2, 3])
```
"""
function sweepcmd(cmd_template, sweeps::Pair...)
    n_sweeps = length(sweeps)
    names = map(s -> s.first isa Symbol ? "@$(s.first)" : s.first, sweeps)
    vlists = map(s -> s.second, sweeps)
    cmds = Cmd[]
    for values in Base.Iterators.product(vlists...)
        cms_str = cmd_template
        for i = 1:n_sweeps
            cms_str = replace(cms_str, names[i] => string(values[i]))
        end
        cmd = Cmd(map(String, split(cms_str, " ")))
        push!(cmds, cmd)
    end
    return cmds
end

"""
    sweepcmd(cmd_template, sweeps)

Run a list of commands given a command template and sweep mappings.

Example:

```julia
sweeprun("sleep @Ts", "@T" => [1, 2, 3])
sweeprun("sleep @Ts", :T => [1, 2, 3])
```
"""
function sweeprun(cmd_template, sweeps::Pair...; maxasync=0)
    cmds = sweepcmd(cmd_template, sweeps...)
    for cmds_part in Iterators.partition(cmds, maxasync > 0 ? maxasync : length(cmds))
        @sync begin
            for cmd in cmds_part
                @async run(cmd)
            end
        end
    end
end

### Logging

function figure_to_image(fig::PyPlot.Figure; close=true)
    canvas = plt_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = canvas.buffer_rgba() ./ 255
    w, h = fig.canvas.get_width_height()
    img = [Images.RGBA(data[r,c,:]...) for r in 1:h, c in 1:w]
    if close plt.close(fig) end
    return img
end

TensorBoardLogger.preprocess(name, fig::PyPlot.Figure, data) = push!(data, name => figure_to_image(fig))

import Logging

"""
Combine multiple loggers into a single logger.
"""
struct CombinedLogger <: Base.CoreLogging.AbstractLogger
    loggers
    min_level::Logging.LogLevel
    message_limits::Dict{Any,Int}
end

# TODO: make sure the logging level operations below are correct.
function CombinedLogger(loggers::Base.CoreLogging.AbstractLogger...)
    min_level = min(map(l -> l.min_level, loggers)...)
    return CombinedLogger(loggers, min_level, Dict{Any,Int}())
end

Logging.shouldlog(logger::CombinedLogger, level, _module, group, id) =
    get(logger.message_limits, id, 1) > 0

Logging.min_enabled_level(logger::CombinedLogger) = logger.min_level

Logging.catch_exceptions(logger::CombinedLogger) = false

function Logging.handle_message(clogger::CombinedLogger, args...; kwargs...)
    for logger in clogger.loggers
        Logging.handle_message(logger, args...; kwargs...)
    end
    nothing
end