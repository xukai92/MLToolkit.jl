abstract type AbstractNeuralModel end

optional_BatchNorm(D, σ, isnorm) = isnorm ? BatchNorm(D, σ) : x -> σ.(x)

### MLP

const IntIte = Union{AbstractVector{Int},Tuple{Vararg{Int}}}

struct MLP
    f
end

Flux.@functor MLP

MLP(args...; kwargs...) = MLP(build_mlp(args...; kwargs...))

(m::MLP)(x::AbstractArray{<:Real,2}) = m.f(x)

function build_mlp(Dhs::IntIte, σs; isnorm::Bool=false)
    @assert length(σs) == length(Dhs) - 1 "Length of `σs` should be greater than the length of `Dhs` by 1."
    layers = []
    for i in 1:length(Dhs)-2
        push!(layers, Dense(Dhs[i], Dhs[i+1]))
        push!(layers, optional_BatchNorm(Dhs[i+1], σs[i], isnorm))
    end
    push!(layers, Dense(Dhs[end-1], Dhs[end], σs[end]))
    return Chain(layers...)
end

build_mlp(Dhs::IntIte, σ::Function, σlast::Function; kwargs...) = build_mlp(Dhs, (fill(σ, length(Dhs) - 2)..., σlast); kwargs...)

build_mlp(Din::Int, Dhs::IntIte, arg...; kwargs...) = build_mlp([Din, Dhs...], arg...; kwargs...)
build_mlp(Dhs::IntIte, Dout::Int, arg...; kwargs...) = build_mlp([Dhs..., Dout], arg...; kwargs...)
build_mlp(Din::Int, Dhs::IntIte, Dout::Int, arg...; kwargs...) = build_mlp([Din, Dhs..., Dout], arg...; kwargs...)
build_mlp(Din::Int, Dout::Int, arg...; kwargs...) = build_mlp([Din, Dout], arg...; kwargs...)

### ConvNet

struct ConvNet <: AbstractNeuralModel
    WHCin::Tuple{Int,Int,Int}
    f
end

Flux.@functor ConvNet

function ConvNet(WHCin::Tuple{Int,Int,Int}, Dout::Int, args...; kwargs...)
    if WHCin == (28, 28, 1)
        f = build_convnet_inmnist(Dout, args...; kwargs...)
    else
        throw(ErrorException("Unsupported input and output size for `build_convnet`: WHCin=$WHCin, Dout=$Dout."))
    end
    return ConvNet(WHCin, f)
end

function (m::ConvNet)(x::AbstractArray{<:Real,4})
    let WHCin=m.WHCin
        WHCin[1] != size(x, 1) && throw(DimensionMismatch("`WHCin[1]` ($(WHCin[1])) != `size(x, 1)` ($(size(x, 1)))"))
        WHCin[2] != size(x, 2) && throw(DimensionMismatch("`WHCin[2]` ($(WHCin[2])) != `size(x, 2)` ($(size(x, 2)))"))
        WHCin[3] != size(x, 3) && throw(DimensionMismatch("`WHCin[3]` ($(WHCin[3])) != `size(x, 3)` ($(size(x, 3)))"))
    end
    return m.f(x)
end

(m::ConvNet)(x::AbstractArray{<:Real,2}) = m(reshape(x, m.WHCin..., size(x, 2)))

function build_convnet_inmnist(Dout::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convnet_inmnist`"
    return Chain(
        #    28 x 28 x  1 x B
        Conv((3, 3), 1 => 16,  pad=(1, 1)), optional_BatchNorm(16, σs[1], isnorm), MaxPool((2, 2)),
        # -> 14 x 14 x 16 x B
        Conv((3, 3), 16 => 32, pad=(1, 1)), optional_BatchNorm(32, σs[2], isnorm), MaxPool((2, 2)),
        # ->  7 x  7 x 32 x B
        Conv((3, 3), 32 => 32, pad=(1, 1)), optional_BatchNorm(32, σs[3], isnorm), MaxPool((2, 2)),
        # ->  3 x  3 x 32 x B
        x -> reshape(x, :, size(x, 4)),
        # ->  288 x B
        Dense(288, Dout, σs[4])
    )
end

build_convnet_inmnist(Dout::Int, σ::Function, σlast::Function; kwargs...) = build_convnet_inmnist(Dout, (σ, σ, σ, σlast); kwargs...)
build_convnet_inmnist(Dout::Int, σ::Function; kwargs...) = build_convnet_inmnist(Dout, σ, σ; kwargs...)

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
