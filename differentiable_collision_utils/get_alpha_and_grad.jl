include("update_arm.jl")
include("diff_opt_link_polygon_obstacle.jl")

function get_alpha_and_grad(rs::Vector{Float64}, qs::Vector{Float64})
    #=
        This function computes the value of
        α and ∂α/∂(r, q) between four polygon 
        obstacles and all of the robot links 
        that are relavant. This function requires 
        all of the links and obstacles be already 
        initialized.
    =#

    # update the position and orientation of all of the links
    update_arm(rs, qs)

    # compute α and J
    αs, Js = diff_opt_link_polygon_obstacle(table)

    return [αs], [Js]
end