export symmetric_cauchy_kernel, symmetric_blob_kernel

# w(ξ) = ∑_J ΓJ k_{symmetric, cauchy}(ξ - z)
"Defines the symmetric Cauchy kernel induced by two mirrored point vortices located at z, conj()"
symmetric_cauchy_kernel(ξ, z) = (ξ - z) != zero(z) ? imag(z)/(π*(ξ - z)*(ξ - conj(z))) : zero(z)

# w(ξ) = ∑_J ΓJ k_{symmetric, blob}(ξ - z, δ)
symmetric_blob_kernel(ξ, z, δ::Float64) = -im/(2*π)*(conj(ξ - z)/(abs2(ξ - z) + δ^2) - (conj(ξ) - z)/(abs2(ξ - conj(z)) + δ^2))
