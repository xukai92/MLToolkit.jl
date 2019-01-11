import ArgParse: parse_args

function parse_args(args_str::AbstractString, settings; as_symbols::Bool=false)
    parse_args(split(args_str, " "), settings; as_symbols=as_symbols)
end
