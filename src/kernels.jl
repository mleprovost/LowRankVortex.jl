export symmetric_cauchy_kernel, symmetric_blob_kernel

# w(ξ) = ∑_J ΓJ k_{symmetric, cauchy}(ξ - z)
"""
Symmetric Cauchy kernel induced at ξ by two mirrored point vortices located at z, conj(z) with strength s and conj(s).
The strength is not included in the result.
"""
symmetric_cauchy_kernel(ξ, z) = (ξ - z) != zero(z) ? imag(z)/(π*(ξ - z)*(ξ - conj(z))) : zero(z)

# w(ξ) = ∑_J ΓJ k_{symmetric, blob}(ξ - z, δ)
"""
Symmetric Blob kernel induced at ξ by two mirrored regularized point vortices located at z, conj(z) with strength s and conj(s) with blob radius δ.
The strength is not included in the result.
"""
symmetric_blob_kernel(ξ, z, δ::Float64) = -im/(2*π)*(conj(ξ - z)/(abs2(ξ - z) + δ^2) - (conj(ξ) - z)/(abs2(ξ - conj(z)) + δ^2))
