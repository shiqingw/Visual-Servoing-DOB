import StaticArrays as sa
import DifferentiableCollisions as dc

function create_obstacle_polygon(table_b_input::Vector{Float64}, table_r_input::Vector{Float64},
     table_q_input::Vector{Float64})
    table_A = sa.@SMatrix [1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
        -1.0 0.0 0.0
        0.0 -1.0 0.0
        0.0 0.0 -1.0]

    table_b = sa.SVector{length(table_b_input), Float64}(table_b_input)

    global table = dc.Polytope(table_A, table_b)

    table.r = sa.SVector{length(table_r_input), Float64}(table_r_input)
    table.q = sa.SVector{length(table_q_input), Float64}(table_q_input)
end