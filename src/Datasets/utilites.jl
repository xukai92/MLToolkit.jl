Base.getindex(arr::AbstractArray{<:Any,3}, ::Colon, i) = arr[:,:,i]
Base.getindex(arr::AbstractArray{<:Any,4}, ::Colon, i) = arr[:,:,:,i]

datadim(X::AbstractMatrix) = size(X, 1)
datadim(X::AbstractArray) = Base.front(size(X))

_getproperty(d::AbstractDataset, ::Val{T}) where {T} = getfield(d, T)
Base.getproperty(d::AbstractDataset, k::Symbol) = _getproperty(d, Val(k))

_getproperty(d::AbstractDataset, ::Val{:dim})       = datadim(d.X)
_getproperty(d::AbstractDataset, ::Val{:n_data})    = last(size(d.X))
_getproperty(d::AbstractDataset, ::Val{:n_test})    = last(size(d.Xt))
_getproperty(d::AbstractDataset, ::Val{:n_display}) = n_display(d)
_getproperty(d::AbstractDataset, ::Val{:x_display}) = d.X[:,1:n_display(d)]
_getproperty(d::AbstractDataset, ::Val{:x_display}) = (args...) -> vis(d, args...)

const DATASET_NAMES = (
    "gaussian", "2dring", "3dring", # simple
    "mnist", "cifar10",             # image
    "griffiths2011", "xu2019",      # feature
)

function Dataset(name::String, n_data::Int, args...; is_preview=true, kwargs...)
    if name == "gaussian"
        dataset = GaussianDataset(n_data, args...; kwargs...)
    end
    if name == "2dring"
        dataset = TwoDimRingDataset(n_data, args...; kwargs...)
    end
    if name == "3dring"
        dataset = ThreeDimRingDataset(n_data, args...; kwargs...)
    end
    if name == "mnist"
        dataset = MNISTDataset(n_data, args...; kwargs...)
    end
    if name == "cifar10"
        dataset = CIFAR10Dataset(n_data, args...; kwargs...)
    end
    if name == "griffiths2011"
        dataset = FeatureDataset(n_data, get_features_griffiths2011(), args...; kwargs...)
    end
    if name == "xu2019"
        dataset = FeatureDataset(n_data, get_features_xu2019(), args...; kwargs...)
    end
    
    if is_preview
        nd_half = div(n_display(dataset), 2)
        fig = vis(
            dataset, 
            (train=dataset.X[:,1:nd_half], test=dataset.Xt[:,1:nd_half])
        )
        display(fig)
        println()
    end
    
    return dataset
end
