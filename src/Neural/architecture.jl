optional_BatchNorm(D, σ, norm) = norm ? BatchNorm(D, σ) : x -> σ.(x)

function build_convnet_inmnist(Dout::Int, σ, σ_last; norm::Bool=false)
    return Chain(
        x -> reshape(x, 28, 28, 1, last(size(x))),
        # -> 28 x 28 x  1
        Conv((3, 3), 1 => 16,  pad=(1, 1)), optional_BatchNorm(16, σ, norm), MaxPool((2, 2)),
        # -> 14 x 14 x 16
        Conv((3, 3), 16 => 32, pad=(1, 1)), optional_BatchNorm(32, σ, norm), MaxPool((2, 2)),
        # ->  7 x  7 x 32
        Conv((3, 3), 32 => 32, pad=(1, 1)), optional_BatchNorm(32, σ, norm), MaxPool((2, 2)),
        # ->  3 x  3 x 32
        x -> reshape(x, :, size(x, 4)),
        Dense(288, Dout, σ_last)
    )
end

# function build_conv_chain(D_in, D_h, D_out, σ, σ_last)
#     return Chain(
#         Dense(D_z, 144, relu),
#         x -> reshape(x, 3, 3, 16, last(size(x))), # ( 3,  3, 16, 1)
#         ConvTranspose((3, 3), 16 => 8, relu),     # ( 5,  5,  8, 1)
#         ConvTranspose((6, 6), 8 => 4, relu),      # (10, 10,  4, 1)
#         ConvTranspose((8, 8), 4 => 2, relu),      # (17, 17,  2, 1)
#         ConvTranspose((12, 12), 2 => 1, sigmoid), # (28, 28,  1, 1)
#         x -> reshape(x, D_x, last(size(x)))
#     )
# end

function build_convnet(Din::Int, Dout::Int, args...; kwargs...)
    if Din == 28 * 28
        return build_convnet_inmnist(Dout, args...; kwargs...)
    else
        @error "Unsupported input and output size for `build_convnet`: Din=$Din, Dout=$Dout."
    end
end

const IntIte = Union{AbstractVector{Int},Tuple{Vararg{Int}}}

function build_mlp(Dlist::IntIte, σ, σ_last; norm::Bool=false)
    layers = []
    for i in 1:length(Dlist)-2
        push!(layers, Dense(Dlist[i], Dlist[i+1]))
        push!(layers, optional_BatchNorm(Dlist[i+1], σ, norm))
    end
    push!(layers, Dense(Dlist[end-1], Dlist[end], σ_last))
    return Chain(layers...)
end

build_mlp(Din::Int, Dlist::IntIte, arg...; kwargs...) = build_mlp([Din, Dlist...], arg...; kwargs...)
build_mlp(Dlist::IntIte, Dout::Int, arg...; kwargs...) = build_mlp([Dlist..., Dout], arg...; kwargs...)
build_mlp(Din::Int, Dlist::IntIte, Dout::Int, arg...; kwargs...) = build_mlp([Din, Dlist..., Dout], arg...; kwargs...)
