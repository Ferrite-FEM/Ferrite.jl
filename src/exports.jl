export
# Interpolations
    Interpolation,
    RefCube,
    RefTetrahedron,
    BubbleEnrichedLagrange,
    CrouzeixRaviart,
    Lagrange,
    DiscontinuousLagrange,
    Serendipity,
    getnbasefunctions,

# Quadrature
    QuadratureRule,
    getweights,
    getpoints,

# FEValues
    CellValues,
    CellScalarValues,
    CellVectorValues,
    FaceValues,
    FaceScalarValues,
    FaceVectorValues,
    reinit!,
    shape_value,
    shape_gradient,
    shape_symmetric_gradient,
    shape_divergence,
    shape_curl,
    function_value,
    function_gradient,
    function_symmetric_gradient,
    function_divergence,
    function_curl,
    spatial_coordinate,
    getnormal,
    getdetJdV,
    getnquadpoints,

# Grid
    Grid,
    DistributedGrid,
    Node,
    Cell,
    Line,
    Line2D,
    Line3D,
    QuadraticLine,
    Triangle,
    QuadraticTriangle,
    Quadrilateral,
    Quadrilateral3D,
    QuadraticQuadrilateral,
    Tetrahedron,
    QuadraticTetrahedron,
    Hexahedron,
    #QuadraticHexahedron,
    CellIndex,
    FaceIndex,
    EdgeIndex,
    VertexIndex,
    ExclusiveTopology,
    getneighborhood,
    faceskeleton,
    getcells,
    getgrid,
    getlocalgrid,
    getglobalgrid,
    getncells,
    getnodes,
    getnnodes,
    getcelltype,
    getcellset,
    getnodeset,
    getfaceset,
    getedgeset,
    getvertexset,
    getcoordinates,
    getcoordinates!,
    getcellsets,
    getnodesets,
    getfacesets,
    getedgesets,
    getvertexsets,
    global_comm,
    vertex_comm,
    onboundary,
    nfaces,
    addnodeset!,
    addfaceset!,
    addedgeset!,
    addvertexset!,
    addcellset!,
    transform!,
    generate_grid,
    compute_vertex_values,
    is_shared_vertex,
    get_shared_vertices,
    get_shared_faces,
    get_shared_edges,

# Grid coloring
    create_coloring,
    ColoringAlgorithm,
    vtk_cell_data_colors,

# Dofs
    DofHandler,
    close!,
    ndofs,
    num_local_true_dofs,
    num_local_dofs,
    num_global_dofs,
    ndofs_per_cell,
    celldofs!,
    celldofs,
    create_sparsity_pattern,
    create_symmetric_sparsity_pattern,
    dof_range,
    renumber!,
    MixedDofHandler,
    FieldHandler,
    Field,
    reshape_to_nodes,
    num_fields,
    getfieldnames,
    dof_range,
    #entity_dofs,

# Constraints
    ConstraintHandler,
    Dirichlet,
    PeriodicDirichlet,
    collect_periodic_faces,
    collect_periodic_faces!,
    PeriodicFacePair,
    AffineConstraint,
    update!,
    apply!,
    apply_rhs!,
    get_rhs_data,
    apply_zero!,
    add!,
    free_dofs,
    ApplyStrategy,

# iterators
    CellIterator,
    UpdateFlags,
    cellid,

# assembly
    start_assemble,
    assemble!,
    end_assemble,

# VTK export
    vtk_grid,
    vtk_point_data,
    vtk_cell_data,
    vtk_nodeset,
    vtk_cellset,
    vtk_save,
    # vtk_shared_vertices,
    # vtk_shared_faces,
    # vtk_shared_edges,
    # vtk_partitioning,

# L2 Projection
    project,
    L2Projector,

# Point Evaluation
    PointEvalHandler,
    get_point_values,
    PointIterator,
    PointLocation,
    PointScalarValues,
    PointVectorValues
