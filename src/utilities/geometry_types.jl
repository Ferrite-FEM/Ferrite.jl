abstract GeoShape
abstract GeoShape_3D <: GeoShape
abstract GeoShape_2D <: GeoShape
abstract GeoShape_1D <: GeoShape

n_dim(::GeoShape_3D) = 3
n_dim(::GeoShape_2D) = 2
n_dim(::GeoShape_1D) = 1

immutable Line <: GeoShape_1D end
immutable Triangle <: GeoShape_2D end
immutable Square <: GeoShape_2D end
immutable Cube <: GeoShape_3D end
immutable Tetrahedra <: GeoShape_3D end