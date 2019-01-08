"""
    load_mnist(mf::Function, tr_sz::Integer, te_sz::Integer; flatten::Bool=true)

Load a subset of MNIST-like dataset.
"""
function load_mnist(mf::Function, tr_sz::Integer, te_sz::Integer; flatten::Bool=true)
    x_tr, y_tr, x_te, y_te = mf()

    if flatten
        x_tr_sub = Matrix{FT}(undef, 28 * 28, tr_sz)
        x_te_sub = Matrix{FT}(undef, 28 * 28, te_sz)

        for i = 1:tr_sz
            x_tr_sub[:,i] = reshape(x_tr[:,:,1,i]', 28 * 28, 1)
        end

        for i = 1:te_sz
            x_te_sub[:,i] = reshape(x_te[:,:,1,i]', 28 * 28, 1)
        end
    else
        x_tr_sub = Array{FT,4}(x_tr[:,:,:,1:tr_sz])
        x_te_sub = Array{FT,4}(x_te[:,:,:,1:te_sz])
    end

    y_tr_sub = Vector{FT}(y_tr)[1:tr_sz]
    y_te_sub = Vector{FT}(y_te)[1:te_sz]

    return x_tr_sub, y_tr_sub, x_te_sub, y_te_sub
end

struct BatchDataLoader
    data::Tuple
    batch_size::Integer
    num_data::Integer
    data_length::Integer
    drop_last::Bool
    num_batchs::Integer
end

function BatchDataLoader(batch_size::Integer, data...; drop_last=false, atype=nothing)
    num_data = length(data)
    data_length = size(first(data), 2)
    for i = 2:num_data
        @assert data_length == last(size(data[i])) "Data lengthes are inconsistent!"
    end
    # Drop the last batch if not as large as batch_size
    num_batchs = (drop_last ? floor : ceil)(Integer, data_length / batch_size)
    # Map to atype if provided
    data = atype == nothing ? data : map(d -> atype(d), data)
    return BatchDataLoader(data, batch_size, num_data, data_length, drop_last, num_batchs)
end

function Base.iterate(bdl::BatchDataLoader, batch_n=0)
    if batch_n >= bdl.num_batchs
        return nothing
    end
    batch_n = batch_n + 1
    i_start = (batch_n - 1) * bdl.batch_size + 1
    i_end = min(batch_n * bdl.batch_size, bdl.data_length)
    if bdl.num_data > 1
        data_batch = map(d -> length(size(d)) == 2 ? d[:,i_start:i_end] : d[i_start:i_end], bdl.data)
    else
        d = first(bdl.data)
        data_batch = length(size(d)) == 2 ? d[:,i_start:i_end] : d[i_start:i_end]
    end
    return data_batch, batch_n
end

Base.length(bdl::BatchDataLoader) = bdl.num_batchs
