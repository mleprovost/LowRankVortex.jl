
@testset "Convective derivative of the complex potential collection of vortices" begin
    atol = 100*eps()
    nblob = 20
    pos = randn(ComplexF64,nblob)
    str = randn(length(pos))
    σ = 1e-4

    blobs = Vortex.Blob.(pos,str, σ)
    blobsvel = randn(ComplexF64, nblob)

    dFdt = complex(0.0)
    for i=1:nblob
        dFdt += circulation(blobs[i])/(2*π*im)*(-blobsvel[i])*1/(0.0 - blobs[i].z)
    end
    @test isapprox(dFdt, convective_complexpotential(complex(0.0), blobs, blobsvel), atol = atol)
end

@testset "Convective derivative of the complex potential collection of sources/sinks" begin
    atol = 100*eps()
    nblob = 20
    pos = randn(ComplexF64,nblob)
    str = randn(length(pos))
    σ = 1e-4

    blobs = Source.Blob.(pos,str, σ)
    blobsvel = randn(ComplexF64, nblob)

    dFdt = complex(0.0)
    for i=1:nblob
        dFdt += flux(blobs[i])/(2*π)*(-blobsvel[i])*1/(0.0 - blobs[i].z)
    end
    @test isapprox(dFdt, convective_complexpotential(complex(0.0), blobs, blobsvel), atol = atol)
end
