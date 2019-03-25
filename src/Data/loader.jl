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
    data_length = last(size(first(data)))
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

function Base.rand(bld::BatchDataLoader)
    i_rand = rand(1:length(bld))
    for (i, data_batch) in enumerate(bld)
        if i == i_rand
            return data_batch
        end
    end
end

Base.length(bdl::BatchDataLoader) = bdl.num_batchs