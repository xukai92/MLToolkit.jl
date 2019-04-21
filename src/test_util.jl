const NUM_RANDTESTS = 5
const ATOL = FT == Float64 ? 1e-6 : 1e-4
const ATOL_RAND = FT == Float64 ? 2e-2 : 1e-1

function include_list_as_module(list, module_name_prefix)
    return Distributed.map(list) do t
        @eval module $(Symbol("$(module_name_prefix)_", t))
            include($t)
        end
        return
    end
end
