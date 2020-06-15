struct LabelledScatter{T<:NamedTuple}
    nt::T
end

LabelledScatter(x) = LabelledScatter((_=x,))

@recipe function f(ls::LabelledScatter)
    @unpack nt = ls
    
    seriesalpha = length(nt) > 0.75 ? 0.5 : 1.0
    
    for (data, label) in zip(values(nt), keys(nt))
        @series begin
            seriestype := :scatter
            markershape := :circle
            seriesalpha := seriesalpha
            label := label
            if size(data, 1) == 2
                data[1,:], data[2,:]
            elseif size(data, 1) == 3
                data[1,:], data[2,:], data[3,:]
            else
                error("Only 2D or 3D scatter are supported.")
            end
        end
    end

    lims --> autoget_lims(first(values(nt)))

    if length(nt) > 1
        legend := :auto
    else
        legend := :none
    end

    nothing
end
