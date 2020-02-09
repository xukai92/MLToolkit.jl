datadim(X::AbstractMatrix) = size(X, 1)
datadim(X::AbstractArray) = Base.front(size(X))

function Base.getproperty(d::AbstractDataset, k::Symbol)
    if k == :dim
        return datadim(d.X)
    end
    if k == :n_data
        return last(size(d.X))
    end
    if k == :n_test
        return last(size(d.Xt))
    end
    if k == :n_display
        return n_display(d)
    end
    if k == :vis
        function _vis(args...)
            return vis(d, args...)
        end
        return _vis
    end
    return getfield(d, k)
end

const DATASET_NAMES = (
    "gaussian",
    "2dring",
    "3dring",
    "mnist",
    "cifar10",
    "griffiths2011",
    "xu2019",
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
    end
    
    return dataset
end
