optional_BatchNorm(D, σ, isnorm; kwargs...) = isnorm ? BatchNorm(D, σ; kwargs...) : x -> σ.(x)

### DenseNet

const IntIte = Union{AbstractVector{Int},Tuple{Vararg{Int}}}

struct DenseNet <: AbstractNeuralModel
    f
end

Flux.@functor DenseNet

DenseNet(args...; kwargs...) = DenseNet(build_densenet(args...; kwargs...))

(m::DenseNet)(x::AbstractMatrix) = m.f(x)

(m::DenseNet)(x::AbstractArray) = m(reshape(x, prod(Base.front(size(x))), last(size(x))))

function build_densenet(Dhs::IntIte, σs; isnorm::Bool=false)
    @assert length(σs) == length(Dhs) - 1 "Length of `σs` should be greater than the length of `Dhs` by 1."
    layers = []
    for i in 1:length(Dhs)-2
        push!(layers, Dense(Dhs[i], Dhs[i+1]))
        push!(layers, optional_BatchNorm(Dhs[i+1], σs[i], isnorm))
    end
    push!(layers, Dense(Dhs[end-1], Dhs[end], σs[end]))
    return Chain(layers...)
end

build_densenet(Dhs::IntIte, σ::Function, σlast::Function=identity; kwargs...) = 
    build_densenet(Dhs, (fill(σ, length(Dhs) - 2)..., σlast); kwargs...)

build_densenet(Din::Int, Dhs::IntIte, arg...; kwargs...) = 
    build_densenet([Din, Dhs...], arg...; kwargs...)
build_densenet(Dhs::IntIte, Dout::Int, arg...; kwargs...) = 
    build_densenet([Dhs..., Dout], arg...; kwargs...)
build_densenet(Din::Int, Dhs::IntIte, Dout::Int, arg...; kwargs...) = 
    build_densenet([Din, Dhs..., Dout], arg...; kwargs...)
build_densenet(Din::Int, Dout::Int, arg...; kwargs...) = 
    build_densenet([Din, Dout], arg...; kwargs...)

### ConvNet

struct ConvNet{Tin<:Union{Int, NTuple{3, Int}}, Tout<:Union{Int, NTuple{3, Int}}} <: AbstractNeuralModel
    Sin::Tin
    f
    Sout::Tout
end

Flux.@functor ConvNet

## Conv in

function ConvNet(WHCin::NTuple{3, Int}, Dout::Int, args...; kwargs...)
    if WHCin == (28, 28, 1)
        f = build_convin_mnist(Dout, args...; kwargs...)
    elseif WHCin == (32, 32, 3)
        f = build_convin_incifar(Dout, args...; kwargs...)
    else
        throw(ErrorException("Unsupported input and output size for `ConvNet`: `WHCin`=$WHCin, `Dout`=$Dout."))
    end
    return ConvNet(WHCin, f, Dout)
end

function (m::ConvNet{NTuple{3, Int}, Int})(x::AbstractArray{<:Real,4})
    let (W, H, C) = m.Sin
        W != size(x, 1) && throw(DimensionMismatch("`Sin[1]` ($W) != `size(x, 1)` ($(size(x, 1)))"))
        H != size(x, 2) && throw(DimensionMismatch("`Sin[2]` ($H) != `size(x, 2)` ($(size(x, 2)))"))
        C != size(x, 3) && throw(DimensionMismatch("`Sin[3]` ($C) != `size(x, 3)` ($(size(x, 3)))"))
    end
    return m.f(x)
end

(m::ConvNet{NTuple{3, Int}, Int})(x::AbstractMatrix) = m(reshape(x, m.Sin..., size(x, 2)))

function build_convin_mnist(Dout::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convin_mnist`"
    return Chain(
        #    28 x 28 x  1 x B
        Conv((3, 3),  1 => 16; pad=(1, 1)), optional_BatchNorm(16, σs[1], isnorm), MaxPool((2, 2)),
        # -> 14 x 14 x 16 x B
        Conv((3, 3), 16 => 32; pad=(1, 1)), optional_BatchNorm(32, σs[2], isnorm), MaxPool((2, 2)),
        # ->  7 x  7 x 32 x B
        Conv((3, 3), 32 => 32; pad=(1, 1)), optional_BatchNorm(32, σs[3], isnorm), MaxPool((2, 2)),
        # ->  3 x  3 x 32 x B
        x -> reshape(x, :, size(x, 4)),
        # ->  288 x B
        Dense(288, Dout, σs[4])
    )
end

build_convin_mnist(Dout::Int, σ::Function, σlast::Function=identity; kwargs...) = 
    build_convin_mnist(Dout, (σ, σ, σ, σlast); kwargs...)


#  function build_convin_incifar(Dout::Int, σs; isnorm::Bool=false)
#      @assert length(σs) == 5 "Length of `σs` must be `5` for `build_convin_incifar`"
#      return Chain(
#          #    32 x 32 x   3 x B
#          Conv((4, 4),  3 =>  32; stride=(2, 2), pad=(1, 1)), optional_BatchNorm( 32, σs[1], isnorm),
#          # -> 16 x 16 x  32 x B
#          Conv((4, 4), 32 =>  64; stride=(2, 2), pad=(1, 1)), optional_BatchNorm( 64, σs[2], isnorm),
#          # ->  8 x  8 x  64 x B
#          Conv((4, 4), 64 => 128; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(128, σs[3], isnorm),
#          # ->  4 x  4 x 128 x B
#          x -> reshape(x, :, size(x, 4)),
#          # -> 2048 x B
#          Dense(2048, 512), optional_BatchNorm(512, σs[4], isnorm),
#          # ->  512 x B
#          Dense(512, Dout, σs[5])
#          # -> Dout x B
#      )
#  end

#  function build_convin_incifar(Dout::Int, σs; isnorm::Bool=false)
#      @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convin_incifar`"
#      return Chain(
#          #    32 x 32 x   3 x B
#          Conv((4, 4),  3 =>  32; pad=(1, 1)), optional_BatchNorm( 32, σs[1], isnorm), MaxPool((2, 2)),
#          # -> 15 x 15 x  32 x B
#          Conv((4, 4), 32 =>  64; pad=(1, 1)), optional_BatchNorm( 64, σs[2], isnorm), MaxPool((2, 2)),
#          # ->  7 x  7 x  64 x B
#          Conv((4, 4), 64 => 128; pad=(1, 1)), optional_BatchNorm(128, σs[3], isnorm), MaxPool((2, 2)),
#          # ->  3 x  3 x 128 x B
#          x -> reshape(x, :, size(x, 4)),
#          # -> 1152 x B
#          Dense(1152, Dout, σs[4])
#          # -> Dout x B
#      )
#  end

# Akash version

function build_convin_incifar(Dout::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convin_incifar`"
    return Chain(
        #    32 x 32 x   3 x B
        Conv((4, 4),  3 =>  32, σs[1]; stride=(2, 2), pad=(1, 1)), 
        # -> 16 x 16 x  32 x B
        Conv((4, 4), 32 =>  64; stride=(2, 2), pad=(1, 1)), optional_BatchNorm( 64, σs[2], isnorm),
        # ->  8 x  8 x  64 x B
        Conv((4, 4), 64 => 128; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(128, σs[3], isnorm),
        # ->  4 x  4 x 128 x B
        x -> reshape(x, :, size(x, 4)),
        # -> 2048 x B
        Dense(2048, Dout, σs[4])
        # -> Dout x B
    )
end

build_convin_incifar(Dout::Int, σ::Function, σlast::Function=identity=identity; kwargs...) = 
    build_convin_incifar(Dout, (σ, σ, σ, σlast); kwargs...)

## Conv out

function ConvNet(Din::Int, WHCout::Tuple{Int,Int,Int}, args...; kwargs...)
    if WHCout == (28, 28, 1)
        f = build_convout_mnist(Din, args...; kwargs...)
    elseif WHCout == (32, 32, 3)
        f = build_convout_cifar(Din, args...; kwargs...)
    else
        throw(ErrorException("Unsupported input and output size for `ConvNet`: `Din`=$Din, `WHCout`=$WHCout."))
    end
    return ConvNet(Din, f, WHCout)
end

(m::ConvNet{Int, NTuple{3, Int}})(x) = m.f(x)

function build_convout_mnist(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 5 "Length of `σs` must be `5` for `build_convout_mnist`"
    return Chain(
        #     Din x B
        Dense(Din,  288), optional_BatchNorm(288, σs[1], isnorm),
        #     288 x B
        Dense(288, 1024),
        # -> 1024 x B
        x -> reshape(x, 4, 4, 64, last(size(x))), optional_BatchNorm(64, σs[2], isnorm),
        # ->    4 x  4 x 64 x B
        ConvTranspose((4, 4), 64 => 32; stride=(1, 1), pad=(0, 0)), optional_BatchNorm(32, σs[3], isnorm),
        # ->    7 x  7 x 32 x B
        ConvTranspose((4, 4), 32 => 16; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(16, σs[4], isnorm),
        # ->   15 x 15 x 16 x B
        ConvTranspose((4, 4), 16 =>  1; stride=(2, 2), pad=(1, 1)), x -> σs[5].(x)
        # ->   28 x 28 x  1 x B
    )
end

build_convout_mnist(Din::Int, σ::Function, σlast::Function=identity; kwargs...) = 
    build_convout_mnist(Din, (σ, σ, σ, σ, σlast); kwargs...)

# FIXME: make it 5 layers
function build_convout_cifar(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convnet_outcifar10`"
    return Chain(
        #     Din x B
        Dense(Din,  2048), 
        # -> 2048 x B
        x -> reshape(x, 4, 4, 128, size(x, 2)), 
        optional_BatchNorm(128, σs[1], isnorm; momentum=9f-1),
        # ->    4 x  4 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), 
        optional_BatchNorm(64, σs[2], isnorm; momentum=9f-1),
        # ->    8 x  8 x  64 x B        
        ConvTranspose((4, 4),  64 => 32; stride=(2, 2), pad=(1, 1)), 
        optional_BatchNorm(32, σs[3], isnorm; momentum=9f-1),
        # ->   16 x 16 x  64 x B
        ConvTranspose((4, 4),  32 =>  3; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # ->   32 x 32 x   3 x B
    )
end

build_convout_cifar(Din::Int, σ::Function, σlast::Function; kwargs...) = 
    build_convout_cifar(Din, (σ, σ, σ, σlast); kwargs...)
