module Neural

import Flux

nparams(m) = sum(prod.(size.(Flux.params(m))))

export nparams

end # module
