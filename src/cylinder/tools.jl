export create_random_vortices

# Create a set of n random point vortices in the range [-2,2]x[-2,2], all of which are outside the unit circle
function create_random_vortices(n::Integer;σ=0.01,cylinder=true)
    z = ComplexF64[]
    Γ = Float64[]
    num = 0
    while num < n
        ztest = 4.0(rand(ComplexF64)-0.5-im*0.5)
        # Γtest = 2.0(rand(Float64)-0.5)
        Γtest = 2.0*rand(Float64)+1.0
        if (!cylinder || abs2(ztest)>=1.0)
            push!(z,ztest)
            push!(Γ,Γtest)
            num += 1
        end
    end
    return Vortex.Blob.(z,Γ,σ)
end
<<<<<<< HEAD
=======


>>>>>>> Fixed most of the bugs in the sensitivity derivatives
