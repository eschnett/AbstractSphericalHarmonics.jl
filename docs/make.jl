# Generate documentation with this command:
# (cd docs && julia --color=yes make.jl)

push!(LOAD_PATH, "..")

using Documenter
using AbstractSphericalHarmonics

makedocs(; sitename="AbstractSphericalHarmonics", format=Documenter.HTML(), modules=[AbstractSphericalHarmonics])

deploydocs(; repo="github.com/eschnett/AbstractSphericalHarmonics.jl.git", devbranch="main", push_preview=true)
