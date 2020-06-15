### TwoDimPath

struct TwoDimPath
    x
    y
end

TwoDimPath(xy::AbstractMatrix) = TwoDimPath(path[1,:], path[2,:])

@recipe function f(path::TwoDimPath, start=(color=:green, marker=:circle), stop=(color=:red, marker=:rect))
    @unpack x, y = path
    
    seriestype --> :path
    seriescolor --> :gray
    label --> :none
    
    @series begin
        seriestype := :scatter
        markershape := start.marker
        seriescolor := start.color
        label := "start"
        [x[1]], [y[1]]
    end
    @series begin
        seriestype := :scatter
        markershape := stop.marker
        seriescolor := stop.color
        label := "stop"
        [x[end]], [y[end]]
    end
    
    x, y
end
