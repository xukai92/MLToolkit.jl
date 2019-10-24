module Neural

nparams(m) = sum(prod.(size.(Flux.params(m))))

export nparams

end # module
