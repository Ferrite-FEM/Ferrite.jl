# # [Reactive Surface](@id tutorial-reactive-surface)
#
# ![](reactive_surface.gif)
#
# *Figure 1*: Reactant concentration field of the Gray-Scott model on the unit sphere.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`reactive_surface.ipynb`](@__NBVIEWER_ROOT_URL__/examples/reactive_surface.ipynb).
#-
#
# ## Introduction
#
# This tutorial gives a quick tutorial on how to assemble and solve time-dependent problems
# on embedded surfaces.
#
# For this showcase we use the well known Gray-Scott model, which is a well-known reaction-diffusion
# system to study pattern formation. The strong form is given by
#
# ```math
#  \begin{aligned}
#    \partial_t r_1 &= \nabla \cdot (D_1 \nabla r_1) - r₁*r₂^2 + F *(1 - r₁) \quad \textbf{x} \in \Omega, \\
#    \partial_t r_2 &= \nabla \cdot (D_2 \nabla r_2) + r₁*r₂^2 - r₂*(F + k ) \quad \textbf{x} \in \Omega,
#  \end{aligned}
# ```
#
# where $r_1$ and $r_2$ are the reaction fields, $D_1$ and $D_2$ the diffusion tensors,
# $k$ is the conversion rate, $F$ is the feed rate and $\Omega$ the domain. Depending on the choice of
# parameters a different pattern can be observed. Please also note that the domain does not have a 
# boundary. The corresponding weak form can be derived as usual.
#
# For simplicity we will solve the problem with the Lie-Troter-Godunov operator splitting technique with
# the classical reaction-diffusion split. In this method we split our problem in two problems, i.e. a heat
# problem and a pointwise reaction problem, and solve them alternatingly to advance in time.
#
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref reactive_surface-plain-program).
#
# First we load Ferrite, and some other packages we need

using Ferrite
using BlockArrays, SparseArrays, LinearAlgebra

# ### Assembly routines
# The following assembly routines are similar to these found in previous tutorials.

function assemble_element_mass!(Me::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Reset to 0
    fill!(Me, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δuᵢ = shape_value(cellvalues, q_point, i)
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                δuⱼ = shape_value(cellvalues, q_point, j)
                ## Add contribution to Ke
                Me[2*i-1, 2*j-1] += (δuᵢ * δuⱼ) * dΩ
                Me[2*i  , 2*j  ] += (δuᵢ * δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_element_diffusion!(De::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Reset to 0
    fill!(De, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            ∇δuᵢ = shape_gradient(cellvalues, q_point, i)
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δuⱼ = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                De[2*i-1, 2*j-1] += 2*0.00008 * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
                De[2*i  , 2*j  ] += 2*0.00004 * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_matrices!(M::SparseMatrixCSC, D::SparseMatrixCSC, cellvalues::CellValues, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cellvalues)

    ## Allocate the element stiffness matrix and element force vector
    Me = zeros(2*n_basefuncs, 2*n_basefuncs)
    De = zeros(2*n_basefuncs, 2*n_basefuncs)

    ## Create an assembler
    M_assembler = start_assemble(M)
    D_assembler = start_assemble(D)
    ## Loop over all cels
    for cell in CellIterator(dh)
        ## Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        ## Compute element contribution
        assemble_element_mass!(Me, cellvalues)
        assemble!(M_assembler, celldofs(cell), Me)

        assemble_element_diffusion!(De, cellvalues)
        assemble!(D_assembler, celldofs(cell), De)
    end
    return nothing
end

# ### Initial condition setup
# Time-dependent problems always need an initial condition from which the time evolution starts.
# In this tutorial we set the concentration of reactant 1 to $1$ and the concentration of reactant
# 2 to $0$ for all nodal dof with associated coordinate $z \leq 0.9$ on the sphere. Since the
# simulation would be pretty boring with a steady-state initial condition, we introduce some
# heterogeneity by setting the dofs associated to top part of the sphere (i.e. dofs with $z > 0.9$
# to store the reactant concentrations of $0.5$ and $0.25$ for the reactants 1 and 2 respectively.
function setup_initial_conditions!(u₀::Vector, cellvalues::CellValues, dh::DofHandler)
    u₀ .= ones(ndofs(dh))
    u₀[2:2:end] .= 0.0

    n_basefuncs = getnbasefunctions(cellvalues)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        coords = getcoordinates(cell)
        dofs = celldofs(cell)
        uₑ = @view u₀[dofs]
        rv₀ₑ = reshape(uₑ, (2, n_basefuncs))

        for i in 1:n_basefuncs            
            if coords[i][3] > 0.9
                rv₀ₑ[1, i] = 0.5
                rv₀ₑ[2, i] = 0.25
            end
        end
    end

    u₀ .+= 0.01*rand(ndofs(dh))
end

# ### Simulation routines
# Now we define a function to setup and solve the problem with given feed and conversion rates
# $F$ and $k$, as well as the time step length and for how long we want to solve the model.
function gray_scott_sphere(F, k, Δt, T)
    ## TODO replace this with a FerriteGmsh reader
    elements = Triangle[Triangle((76, 129, 111)), Triangle((111, 138, 76)), Triangle((91, 117, 116)), Triangle((45, 114, 109)), Triangle((34, 122, 48)), Triangle((45, 109, 46)), Triangle((116, 117, 58)), Triangle((100, 146, 2)), Triangle((5, 21, 18)), Triangle((46, 160, 98)), Triangle((71, 146, 99)), Triangle((27, 122, 34)), Triangle((49, 102, 30)), Triangle((38, 125, 124)), Triangle((12, 146, 71)), Triangle((44, 103, 45)), Triangle((124, 125, 113)), Triangle((98, 126, 46)), Triangle((4, 21, 5)), Triangle((80, 134, 107)), Triangle((20, 36, 13)), Triangle((112, 160, 46)), Triangle((12, 21, 4)), Triangle((80, 114, 103)), Triangle((21, 121, 18)), Triangle((48, 122, 102)), Triangle((109, 112, 46)), Triangle((10, 20, 13)), Triangle((3, 12, 4)), Triangle((44, 121, 103)), Triangle((86, 148, 108)), Triangle((88, 117, 91)), Triangle((99, 146, 100)), Triangle((109, 114, 96)), Triangle((103, 114, 45)), Triangle((110, 148, 86)), Triangle((80, 140, 114)), Triangle((140, 145, 96)), Triangle((99, 134, 71)), Triangle((105, 147, 87)), Triangle((93, 151, 104)), Triangle((76, 157, 89)), Triangle((49, 117, 102)), Triangle((102, 117, 88)), Triangle((90, 138, 111)), Triangle((138, 157, 76)), Triangle((96, 145, 95)), Triangle((102, 122, 30)), Triangle((36, 37, 13)), Triangle((108, 128, 86)), Triangle((131, 141, 130)), Triangle((82, 127, 60)), Triangle((106, 156, 77)), Triangle((24, 124, 113)), Triangle((8, 118, 22)), Triangle((77, 155, 106)), Triangle((27, 137, 122)), Triangle((37, 143, 13)), Triangle((36, 43, 37)), Triangle((108, 148, 59)), Triangle((22, 118, 25)), Triangle((87, 154, 105)), Triangle((36, 98, 43)), Triangle((43, 51, 37)), Triangle((7, 14, 6)), Triangle((12, 33, 21)), Triangle((51, 94, 37)), Triangle((43, 52, 51)), Triangle((12, 71, 33)), Triangle((52, 53, 51)), Triangle((71, 80, 33)), Triangle((52, 55, 53)), Triangle((80, 103, 33)), Triangle((55, 56, 53)), Triangle((56, 60, 53)), Triangle((55, 57, 56)), Triangle((55, 81, 57)), Triangle((56, 61, 60)), Triangle((81, 95, 57)), Triangle((61, 62, 60)), Triangle((81, 96, 95)), Triangle((95, 106, 57)), Triangle((62, 82, 60)), Triangle((61, 63, 62)), Triangle((61, 64, 63)), Triangle((81, 109, 96)), Triangle((62, 87, 82)), Triangle((64, 65, 63)), Triangle((87, 101, 82)), Triangle((64, 66, 65)), Triangle((16, 28, 3)), Triangle((66, 67, 65)), Triangle((4, 16, 3)), Triangle((79, 92, 78)), Triangle((16, 35, 28)), Triangle((66, 68, 67)), Triangle((67, 69, 65)), Triangle((77, 78, 68)), Triangle((92, 93, 78)), Triangle((66, 77, 68)), Triangle((77, 79, 78)), Triangle((39, 40, 35)), Triangle((40, 59, 35)), Triangle((79, 107, 92)), Triangle((74, 83, 73)), Triangle((16, 39, 35)), Triangle((67, 70, 69)), Triangle((72, 75, 69)), Triangle((70, 74, 73)), Triangle((92, 99, 93)), Triangle((59, 130, 35)), Triangle((10, 29, 15)), Triangle((40, 108, 59)), Triangle((10, 15, 9)), Triangle((107, 134, 92)), Triangle((70, 72, 69)), Triangle((85, 131, 83)), Triangle((74, 104, 85)), Triangle((99, 100, 93)), Triangle((70, 73, 72)), Triangle((5, 18, 6)), Triangle((8, 22, 7)), Triangle((8, 17, 9)), Triangle((9, 20, 10)), Triangle((1, 19, 11)), Triangle((74, 85, 83)), Triangle((110, 111, 84)), Triangle((97, 110, 84)), Triangle((89, 105, 75)), Triangle((83, 84, 73)), Triangle((72, 76, 75)), Triangle((83, 97, 84)), Triangle((76, 89, 75)), Triangle((88, 91, 90)), Triangle((48, 102, 88)), Triangle((88, 90, 54)), Triangle((48, 88, 54)), Triangle((54, 86, 47)), Triangle((48, 54, 47)), Triangle((34, 48, 47)), Triangle((34, 47, 38)), Triangle((34, 38, 26)), Triangle((45, 46, 42)), Triangle((27, 34, 26)), Triangle((44, 45, 42)), Triangle((25, 27, 26)), Triangle((41, 44, 42)), Triangle((50, 58, 49)), Triangle((25, 26, 23)), Triangle((41, 42, 32)), Triangle((16, 119, 39)), Triangle((19, 50, 49)), Triangle((22, 25, 23)), Triangle((23, 24, 14)), Triangle((7, 22, 14)), Triangle((104, 153, 93)), Triangle((31, 41, 32)), Triangle((32, 123, 17)), Triangle((60, 127, 53)), Triangle((19, 49, 30)), Triangle((22, 23, 14)), Triangle((103, 121, 33)), Triangle((107, 140, 80)), Triangle((86, 142, 110)), Triangle((3, 28, 2)), Triangle((11, 29, 10)), Triangle((17, 20, 9)), Triangle((130, 161, 131)), Triangle((24, 136, 14)), Triangle((70, 133, 74)), Triangle((31, 32, 17)), Triangle((46, 126, 42)), Triangle((141, 152, 130)), Triangle((100, 152, 141)), Triangle((30, 137, 29)), Triangle((100, 151, 93)), Triangle((35, 130, 28)), Triangle((83, 131, 97)), Triangle((92, 134, 99)), Triangle((148, 161, 59)), Triangle((8, 31, 17)), Triangle((59, 161, 130)), Triangle((75, 132, 69)), Triangle((94, 143, 37)), Triangle((87, 115, 101)), Triangle((20, 123, 36)), Triangle((67, 133, 70)), Triangle((74, 133, 104)), Triangle((86, 128, 47)), Triangle((42, 126, 32)), Triangle((7, 31, 8)), Triangle((33, 121, 21)), Triangle((1, 50, 19)), Triangle((19, 30, 29)), Triangle((19, 29, 11)), Triangle((23, 124, 24)), Triangle((69, 132, 65)), Triangle((105, 132, 75)), Triangle((14, 136, 6)), Triangle((26, 124, 23)), Triangle((106, 135, 57)), Triangle((47, 128, 38)), Triangle((9, 118, 8)), Triangle((57, 135, 56)), Triangle((18, 139, 6)), Triangle((111, 129, 84)), Triangle((116, 157, 91)), Triangle((38, 124, 26)), Triangle((17, 123, 20)), Triangle((4, 119, 16)), Triangle((71, 134, 80)), Triangle((122, 137, 30)), Triangle((72, 129, 76)), Triangle((84, 129, 73)), Triangle((81, 158, 109)), Triangle((82, 149, 127)), Triangle((109, 158, 112)), Triangle((114, 140, 96)), Triangle((10, 13, 11)), Triangle((73, 129, 72)), Triangle((94, 144, 143)), Triangle((89, 147, 105)), Triangle((56, 135, 61)), Triangle((31, 139, 41)), Triangle((39, 113, 40)), Triangle((85, 141, 131)), Triangle((121, 159, 18)), Triangle((61, 135, 64)), Triangle((36, 123, 98)), Triangle((115, 116, 58)), Triangle((65, 132, 63)), Triangle((32, 126, 123)), Triangle((3, 146, 12)), Triangle((29, 137, 15)), Triangle((127, 149, 94)), Triangle((113, 120, 24)), Triangle((68, 133, 67)), Triangle((40, 125, 108)), Triangle((54, 142, 86)), Triangle((144, 162, 50)), Triangle((7, 139, 31)), Triangle((53, 127, 51)), Triangle((93, 153, 78)), Triangle((6, 136, 5)), Triangle((38, 128, 125)), Triangle((15, 118, 9)), Triangle((111, 142, 90)), Triangle((6, 139, 7)), Triangle((91, 138, 90)), Triangle((66, 155, 77)), Triangle((5, 119, 4)), Triangle((77, 156, 79)), Triangle((39, 120, 113)), Triangle((58, 117, 49)), Triangle((119, 120, 39)), Triangle((101, 162, 144)), Triangle((13, 143, 11)), Triangle((105, 154, 132)), Triangle((44, 159, 121)), Triangle((110, 142, 111)), Triangle((132, 154, 63)), Triangle((116, 147, 89)), Triangle((43, 160, 52)), Triangle((52, 160, 112)), Triangle((90, 142, 54)), Triangle((25, 150, 27)), Triangle((52, 158, 55)), Triangle((51, 127, 94)), Triangle((97, 148, 110)), Triangle((5, 136, 119)), Triangle((106, 155, 135)), Triangle((62, 154, 87)), Triangle((78, 153, 68)), Triangle((135, 155, 64)), Triangle((112, 158, 52)), Triangle((107, 145, 140)), Triangle((58, 162, 115)), Triangle((1, 144, 50)), Triangle((79, 145, 107)), Triangle((11, 143, 1)), Triangle((27, 150, 137)), Triangle((95, 156, 106)), Triangle((18, 159, 139)), Triangle((87, 147, 115)), Triangle((113, 125, 40)), Triangle((28, 152, 2)), Triangle((104, 151, 85)), Triangle((64, 155, 66)), Triangle((118, 150, 25)), Triangle((68, 153, 133)), Triangle((55, 158, 81)), Triangle((63, 154, 62)), Triangle((41, 159, 44)), Triangle((119, 136, 120)), Triangle((139, 159, 41)), Triangle((89, 157, 116)), Triangle((94, 149, 144)), Triangle((115, 162, 101)), Triangle((98, 160, 43)), Triangle((2, 146, 3)), Triangle((101, 149, 82)), Triangle((50, 162, 58)), Triangle((125, 128, 108)), Triangle((133, 153, 104)), Triangle((120, 136, 24)), Triangle((2, 152, 100)), Triangle((141, 151, 100)), Triangle((130, 152, 28)), Triangle((143, 144, 1)), Triangle((123, 126, 98)), Triangle((97, 161, 148)), Triangle((145, 156, 95)), Triangle((15, 150, 118)), Triangle((131, 161, 97)), Triangle((91, 157, 138)), Triangle((115, 147, 116)), Triangle((85, 151, 141)), Triangle((79, 156, 145)), Triangle((137, 150, 15)), Triangle((144, 149, 101))]
    nodes = Node{3, Float64}[Node{3, Float64}(Vec{3, Float64}([6.123233995736766e-17, -1.499759782661858e-32, 1.0])), Node{3, Float64}(Vec{3, Float64}([6.123233995736766e-17, -1.499759782661858e-32, -1.0])), Node{3, Float64}(Vec{3, Float64}([0.3090169943749472, -7.568733460868295e-17, -0.9510565162951536])), Node{3, Float64}(Vec{3, Float64}([0.5877852522924729, -1.439658655611993e-16, -0.8090169943749476])), Node{3, Float64}(Vec{3, Float64}([0.8090169943749473, -1.981520145234183e-16, -0.5877852522924734])), Node{3, Float64}(Vec{3, Float64}([0.9510565162951535, -2.329416636978185e-16, -0.3090169943749476])), Node{3, Float64}(Vec{3, Float64}([1.0, -2.449293598294706e-16, -2.449293598294706e-16])), Node{3, Float64}(Vec{3, Float64}([0.9510565162951536, -2.329416636978185e-16, 0.3090169943749472])), Node{3, Float64}(Vec{3, Float64}([0.8090169943749476, -1.981520145234184e-16, 0.5877852522924729])), Node{3, Float64}(Vec{3, Float64}([0.5877852522924734, -1.439658655611994e-16, 0.8090169943749472])), Node{3, Float64}(Vec{3, Float64}([0.3090169943749477, -7.568733460868307e-17, 0.9510565162951535])), Node{3, Float64}(Vec{3, Float64}([0.3569270192488652, 0.280553875011349, -0.8910065241883678])), Node{3, Float64}(Vec{3, Float64}([0.4371529102920968, 0.2734329305187583, 0.8568148957219838])), Node{3, Float64}(Vec{3, Float64}([0.9509980539041696, -0.26748511456887, -0.1550948579243064])), Node{3, Float64}(Vec{3, Float64}([0.7010326263701245, -0.2642085807282103, 0.66237986279339])), Node{3, Float64}(Vec{3, Float64}([0.4557230592119721, -0.2804055822079268, -0.8448012800470386])), Node{3, Float64}(Vec{3, Float64}([0.8579604780797957, 0.2698172279020973, 0.4371526982420629])), Node{3, Float64}(Vec{3, Float64}([0.8579604780797949, 0.2698172279020994, -0.4371526982420629])), Node{3, Float64}(Vec{3, Float64}([0.1506325540503649, -0.2698172279019947, 0.9510565162951921])), Node{3, Float64}(Vec{3, Float64}([0.6808812905078324, 0.2698172279020885, 0.6808812905078397])), Node{3, Float64}(Vec{3, Float64}([0.6212581076243342, 0.2996097381118902, -0.7240665498001932])), Node{3, Float64}(Vec{3, Float64}([0.9503583486325037, -0.269666382128176, 0.1552386921305421])), Node{3, Float64}(Vec{3, Float64}([0.852583363144894, -0.5225860288129365, -0.00233481863673884])), Node{3, Float64}(Vec{3, Float64}([0.8168856169729826, -0.4993851513833295, -0.2886388043221473])), Node{3, Float64}(Vec{3, Float64}([0.810365322419831, -0.5037579254278574, 0.2992256619810952])), Node{3, Float64}(Vec{3, Float64}([0.6692191928879879, -0.7291939860450095, 0.1429048725066452])), Node{3, Float64}(Vec{3, Float64}([0.5982328689380234, -0.6710480447953684, 0.4379634186761435])), Node{3, Float64}(Vec{3, Float64}([0.1396663028336856, -0.2630486925403432, -0.95461966730504])), Node{3, Float64}(Vec{3, Float64}([0.4514631740141711, -0.2764825965481388, 0.8483739601820941])), Node{3, Float64}(Vec{3, Float64}([0.2656805088149847, -0.5189327480995083, 0.812479335236105])), Node{3, Float64}(Vec{3, Float64}([0.9562426915941572, 0.2605200766244171, 0.1331510587579743])), Node{3, Float64}(Vec{3, Float64}([0.8205353904539959, 0.5142067451245198, 0.2496259126792635])), Node{3, Float64}(Vec{3, Float64}([0.4315330339435517, 0.5487851291736285, -0.7159707554176757])), Node{3, Float64}(Vec{3, Float64}([0.4227433703980156, -0.8638491071672681, 0.2739575931249943])), Node{3, Float64}(Vec{3, Float64}([0.2533321381291593, -0.5194986426167724, -0.816053912502256])), Node{3, Float64}(Vec{3, Float64}([0.4746165702604443, 0.5196832012319649, 0.7104002263453397])), Node{3, Float64}(Vec{3, Float64}([0.1794310196242874, 0.4693963227069368, 0.86456439981403])), Node{3, Float64}(Vec{3, Float64}([0.4549014889623809, -0.8896283518779415, -0.04032407314181124])), Node{3, Float64}(Vec{3, Float64}([0.5334972857945512, -0.5064666913889904, -0.6774010160631116])), Node{3, Float64}(Vec{3, Float64}([0.3299386131026945, -0.7304111183526599, -0.5980301913538204])), Node{3, Float64}(Vec{3, Float64}([0.8702062666223009, 0.4902724104444173, -0.0487238862191618])), Node{3, Float64}(Vec{3, Float64}([0.6985731805597851, 0.7142697866770318, 0.04259323001291791])), Node{3, Float64}(Vec{3, Float64}([0.2296843115287244, 0.7082599887008079, 0.6675424371851632])), Node{3, Float64}(Vec{3, Float64}([0.7069936913703737, 0.6602762528044178, -0.25336769791165])), Node{3, Float64}(Vec{3, Float64}([0.5046140344474411, 0.847753185318715, -0.1633377268748682])), Node{3, Float64}(Vec{3, Float64}([0.4459016531317898, 0.8671684464372166, 0.2217895426705266])), Node{3, Float64}(Vec{3, Float64}([0.1957393008348495, -0.9778799588436223, 0.07373542025837701])), Node{3, Float64}(Vec{3, Float64}([0.132948405405323, -0.9163556806890073, 0.377646644323987])), Node{3, Float64}(Vec{3, Float64}([-0.04078294352825544, -0.5000378079804607, 0.8650427400465639])), Node{3, Float64}(Vec{3, Float64}([-0.15879096852963, -0.2269691626539187, 0.9608696204572195])), Node{3, Float64}(Vec{3, Float64}([-0.0627822729549371, 0.6521448166384469, 0.7554902542946654])), Node{3, Float64}(Vec{3, Float64}([-0.00810393003187733, 0.8557402770446653, 0.5173421542476102])), Node{3, Float64}(Vec{3, Float64}([-0.303927994070528, 0.7442868971468831, 0.5946972247755409])), Node{3, Float64}(Vec{3, Float64}([-0.1005857430240445, -0.9805320120393182, 0.1686400950735836])), Node{3, Float64}(Vec{3, Float64}([-0.250312169949072, 0.9087344998521109, 0.3339841708134095])), Node{3, Float64}(Vec{3, Float64}([-0.5249866990376768, 0.7644318034931047, 0.3742098123269927])), Node{3, Float64}(Vec{3, Float64}([-0.4519233842103236, 0.8869049507643479, 0.09573329156347832])), Node{3, Float64}(Vec{3, Float64}([-0.3496351647243795, -0.4195763861086914, 0.8376818655122756])), Node{3, Float64}(Vec{3, Float64}([0.03514175063573507, -0.712088294611358, -0.7012098958512663])), Node{3, Float64}(Vec{3, Float64}([-0.5578090220966426, 0.5621381566949251, 0.6106142707595517])), Node{3, Float64}(Vec{3, Float64}([-0.7517046929582519, 0.5490049396020039, 0.3654225374510175])), Node{3, Float64}(Vec{3, Float64}([-0.7516770591270084, 0.3188376499659225, 0.5773423176473207])), Node{3, Float64}(Vec{3, Float64}([-0.9078594242555867, 0.2839340396968504, 0.308500772919197])), Node{3, Float64}(Vec{3, Float64}([-0.8729743723417194, 0.4810256495227656, 0.08080884689056374])), Node{3, Float64}(Vec{3, Float64}([-0.9781767908615765, 0.2072992377015003, 0.01404250220312223])), Node{3, Float64}(Vec{3, Float64}([-0.8971795561961635, 0.3901597641193664, -0.2069884112842441])), Node{3, Float64}(Vec{3, Float64}([-0.9548315645414797, 0.1004649748246536, -0.2796488372740924])), Node{3, Float64}(Vec{3, Float64}([-0.8206862560040664, 0.2795690009485128, -0.4983123949037174])), Node{3, Float64}(Vec{3, Float64}([-0.9944324272272498, -0.09795994042506151, -0.03883294671846241])), Node{3, Float64}(Vec{3, Float64}([-0.9222459365616031, -0.2071687888868163, -0.3264100571471125])), Node{3, Float64}(Vec{3, Float64}([0.1834719611680492, 0.5010291323794294, -0.8457587409966664])), Node{3, Float64}(Vec{3, Float64}([-0.91361990091357, -0.3989235422328132, -0.07847728401966728])), Node{3, Float64}(Vec{3, Float64}([-0.7955325746014653, -0.491919372526284, -0.3537559238815812])), Node{3, Float64}(Vec{3, Float64}([-0.7611350028438446, -0.2826413132177686, -0.5837699851040954])), Node{3, Float64}(Vec{3, Float64}([-0.9376616866552163, -0.2782070648190273, 0.2083060019866838])), Node{3, Float64}(Vec{3, Float64}([-0.8105616108825711, -0.5611756084770717, 0.167546445542243])), Node{3, Float64}(Vec{3, Float64}([-0.7138470175755108, 0.5780614864020698, -0.3953066574662994])), Node{3, Float64}(Vec{3, Float64}([-0.6051533012954252, 0.4204977029517603, -0.6759964228777704])), Node{3, Float64}(Vec{3, Float64}([-0.464381110892345, 0.6786288794782291, -0.5690457167789902])), Node{3, Float64}(Vec{3, Float64}([0.1877443329604412, 0.734160027560642, -0.6525037313099415])), Node{3, Float64}(Vec{3, Float64}([-0.170160273810633, 0.9844745574140952, 0.04307350718262549])), Node{3, Float64}(Vec{3, Float64}([-0.5249321356083022, 0.3098710149269829, 0.7927333770655985])), Node{3, Float64}(Vec{3, Float64}([-0.5953716525426767, -0.5479615533534468, -0.5875974228969015])), Node{3, Float64}(Vec{3, Float64}([-0.5948775141882189, -0.731153543534381, -0.3339689190484101])), Node{3, Float64}(Vec{3, Float64}([-0.5390479091791861, -0.3243867630844813, -0.7773034025045308])), Node{3, Float64}(Vec{3, Float64}([-0.0520370919307315, -0.9844854839200972, -0.1675722919041359])), Node{3, Float64}(Vec{3, Float64}([-0.6990561222536326, 0.04743990203771749, 0.7134914110445663])), Node{3, Float64}(Vec{3, Float64}([-0.1709321209591976, -0.8735796289436984, 0.4556762468233124])), Node{3, Float64}(Vec{3, Float64}([-0.7974897652646338, -0.4017982288808797, 0.4500758353503858])), Node{3, Float64}(Vec{3, Float64}([-0.3862300653530193, -0.892831085675539, 0.2316872656596398])), Node{3, Float64}(Vec{3, Float64}([-0.4483350138760702, -0.7375533246169005, 0.5049859489919507])), Node{3, Float64}(Vec{3, Float64}([-0.3297241885467164, 0.509198624068942, -0.7949834719876292])), Node{3, Float64}(Vec{3, Float64}([-0.5006088199586549, 0.2260636929674356, -0.8356350974567357])), Node{3, Float64}(Vec{3, Float64}([-0.114341460189287, 0.3765549448176345, 0.9193108310115626])), Node{3, Float64}(Vec{3, Float64}([-0.3640768444247161, 0.9138220136269561, -0.1799371522629953])), Node{3, Float64}(Vec{3, Float64}([-0.07438705466455738, 0.9677528189890058, -0.2406679194183651])), Node{3, Float64}(Vec{3, Float64}([-0.3648584408113792, -0.7551083151164547, -0.5446923449165394])), Node{3, Float64}(Vec{3, Float64}([0.4869006321121222, 0.7190276434148181, 0.4959102967817358])), Node{3, Float64}(Vec{3, Float64}([-0.1521295446889728, 0.2979596432036986, -0.9423781898233057])), Node{3, Float64}(Vec{3, Float64}([-0.2544190872670957, 0.003318804331989557, -0.9670883690604412])), Node{3, Float64}(Vec{3, Float64}([-0.4574608311337778, 0.04016692189285442, 0.8883221298403223])), Node{3, Float64}(Vec{3, Float64}([0.05162779183626617, -0.7587722997293148, 0.6493066827575379])), Node{3, Float64}(Vec{3, Float64}([0.4718555144772151, 0.7467039305053169, -0.4688129836356428])), Node{3, Float64}(Vec{3, Float64}([-0.6603962997927848, -0.03746342128023257, -0.7499821459781414])), Node{3, Float64}(Vec{3, Float64}([-0.8755766085628751, -0.1265053019308967, 0.4662209895756586])), Node{3, Float64}(Vec{3, Float64}([-0.6181791088052646, 0.7728275785374294, -0.1434995584965406])), Node{3, Float64}(Vec{3, Float64}([-0.1690313656792333, 0.7375323982092609, -0.6538152330806537])), Node{3, Float64}(Vec{3, Float64}([0.1200196207970379, -0.8918575867551481, -0.436102437015684])), Node{3, Float64}(Vec{3, Float64}([0.1687644519228687, 0.9855263372163161, 0.01601244578962911])), Node{3, Float64}(Vec{3, Float64}([-0.3483879170786434, -0.8995293935105305, -0.2635767998974518])), Node{3, Float64}(Vec{3, Float64}([-0.5764928164425412, -0.8170915372428317, -0.004177601739351213])), Node{3, Float64}(Vec{3, Float64}([0.1882506182999614, 0.9293683913062556, 0.3175470011675514])), Node{3, Float64}(Vec{3, Float64}([0.6134454888565073, -0.6587663452922019, -0.4355588760569071])), Node{3, Float64}(Vec{3, Float64}([0.2249597314763762, 0.912022857110095, -0.3429393930752427])), Node{3, Float64}(Vec{3, Float64}([-0.5648047232459188, -0.2119475694624456, 0.7975423828224195])), Node{3, Float64}(Vec{3, Float64}([-0.5794573587738784, -0.4650165308383579, 0.6693196511457432])), Node{3, Float64}(Vec{3, Float64}([-0.2366091774971593, -0.673742809876049, 0.7000619424482694])), Node{3, Float64}(Vec{3, Float64}([0.8527104539816098, -0.2459350607989368, 0.4608696426759959])), Node{3, Float64}(Vec{3, Float64}([0.6960448696791582, -0.2371008564340713, -0.6777202396797322])), Node{3, Float64}(Vec{3, Float64}([0.7383552949632746, -0.4356623338363972, -0.5148104401388143])), Node{3, Float64}(Vec{3, Float64}([0.6972157319333021, 0.5281952846001349, -0.4846647959888281])), Node{3, Float64}(Vec{3, Float64}([0.347292924577952, -0.7337483028081979, 0.5839529541531424])), Node{3, Float64}(Vec{3, Float64}([0.6950102695017686, 0.5301053528082009, 0.4857458596953471])), Node{3, Float64}(Vec{3, Float64}([0.6829752857073232, -0.715076601189415, -0.1490309147270943])), Node{3, Float64}(Vec{3, Float64}([0.4146754880735923, -0.8482261971288859, -0.3294792225546194])), Node{3, Float64}(Vec{3, Float64}([0.6438850005135908, 0.699438384671586, 0.3101581083280197])), Node{3, Float64}(Vec{3, Float64}([-0.297254111478026, 0.5256842530929838, 0.7970546149790998])), Node{3, Float64}(Vec{3, Float64}([0.2379496082268838, -0.9521921223690982, -0.1915989197331362])), Node{3, Float64}(Vec{3, Float64}([-0.7701004523871441, -0.6262692692633408, -0.1213758444233331])), Node{3, Float64}(Vec{3, Float64}([-0.04870127384665793, -0.4702806835042472, -0.8811720970664533])), Node{3, Float64}(Vec{3, Float64}([-0.3330278774171408, -0.5614481759300539, -0.7575410078720815])), Node{3, Float64}(Vec{3, Float64}([-0.9689297499821217, 0.02104148294224663, 0.2464394359577509])), Node{3, Float64}(Vec{3, Float64}([-0.846843367254934, 0.005714507259744244, -0.5318116731918389])), Node{3, Float64}(Vec{3, Float64}([-0.04746359565270164, 0.5900070474485388, -0.8060017934525794])), Node{3, Float64}(Vec{3, Float64}([-0.6986480461733326, 0.7071703312001933, 0.1086325469111062])), Node{3, Float64}(Vec{3, Float64}([0.8661092531030132, -0.2422987088950967, -0.4372025815993196])), Node{3, Float64}(Vec{3, Float64}([0.5402416852612612, -0.4857275826755379, 0.6871736584985276])), Node{3, Float64}(Vec{3, Float64}([-0.5959480054569645, -0.7520023710628726, 0.2816707452109351])), Node{3, Float64}(Vec{3, Float64}([0.952973978467038, 0.2564234474475258, -0.1615165996541267])), Node{3, Float64}(Vec{3, Float64}([-0.01462904509434265, 0.877618127075472, -0.4791371558000571])), Node{3, Float64}(Vec{3, Float64}([-0.3034152571707645, -0.286217973431629, -0.9088555734552604])), Node{3, Float64}(Vec{3, Float64}([-0.3017858446842533, -0.953352179764715, -0.006702632772165666])), Node{3, Float64}(Vec{3, Float64}([0.09058624877843947, 0.1905528950128461, 0.9774884785686602])), Node{3, Float64}(Vec{3, Float64}([-0.1973201026088886, 0.08799498112396965, 0.9763819234313025])), Node{3, Float64}(Vec{3, Float64}([-0.2878703617800117, 0.8529298840108065, -0.4354782058495664])), Node{3, Float64}(Vec{3, Float64}([0.149818136421478, 0.3047344838543294, -0.9405803635783765])), Node{3, Float64}(Vec{3, Float64}([-0.7314557619932112, -0.2478245862327181, 0.6352601378219105])), Node{3, Float64}(Vec{3, Float64}([-0.1375827891332612, -0.8707453819315426, -0.4720947531790674])), Node{3, Float64}(Vec{3, Float64}([-0.3433904648122909, 0.2222815344623098, 0.9125096756271025])), Node{3, Float64}(Vec{3, Float64}([0.7251853405213533, -0.4530639739871879, 0.5184971141364874])), Node{3, Float64}(Vec{3, Float64}([-0.4601950358024463, -0.1083766469907528, -0.8811782063861016])), Node{3, Float64}(Vec{3, Float64}([-0.0502277688492161, -0.2892680239368302, -0.9559294856651881])), Node{3, Float64}(Vec{3, Float64}([-0.7104189493493915, 0.1859998577119419, -0.6787554562111777])), Node{3, Float64}(Vec{3, Float64}([-0.8606326124626713, 0.126065665070423, 0.493375064688143])), Node{3, Float64}(Vec{3, Float64}([-0.7897543446414333, 0.6004310913334062, -0.1255809686224593])), Node{3, Float64}(Vec{3, Float64}([-0.507050404155998, 0.7824991382674806, -0.3613792831027939])), Node{3, Float64}(Vec{3, Float64}([-0.6438413033196639, -0.602298941313687, 0.471915629569575])), Node{3, Float64}(Vec{3, Float64}([-0.03553269700960936, 0.9642933100142027, 0.2624420692363861])), Node{3, Float64}(Vec{3, Float64}([0.8430492816075711, 0.4544336531682183, -0.2876768389167709])), Node{3, Float64}(Vec{3, Float64}([0.2860415446647247, 0.828462800035671, 0.4814868883810587])), Node{3, Float64}(Vec{3, Float64}([-0.1824585577315876, -0.7135091380548476, -0.6764714218817652])), Node{3, Float64}(Vec{3, Float64}([-0.3917812796843447, -0.2012538866168797, 0.8977774234243683]))]

    ## We start by setting up grid, dof handler and the matrices for the heat problem.
    grid = Grid(elements, nodes);

    ip = Lagrange{RefTriangle, 1}()
    ip_geo = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    cellvalues = CellValues(qr, ip, ip_geo^3);

    dh = DofHandler(grid);
    add!(dh, :reactants, ip^2);
    close!(dh);

    M = create_sparsity_pattern(dh; coupling=[true false;false true])
    D = create_sparsity_pattern(dh; coupling=[true false;false true])
    assemble_matrices!(M, D, cellvalues, dh);

    ## Since the heat problem is linear and has no time dependent parameters, we precompute the
    ## decomposition of the system matrix to speed up the linear system solver.
    A = M + Δt .* D
    Alu = cholesky(A)

    ## Now we setup buffers for the time dependent solution and fill the initial condition.
    uₜ   = zeros(ndofs(dh))
    uₜ₋₁ = ones(ndofs(dh))
    setup_initial_conditions!(uₜ₋₁, cellvalues, dh)

    ## And prepare output for visualization.
    pvd = paraview_collection("reactive-surface.pvd");
    vtk_grid("reactive-surface-0.0.vtu", dh) do vtk
        vtk_point_data(vtk, dh, uₜ₋₁)
        vtk_save(vtk)
        pvd[0.0] = vtk
    end

    ## This is now the main solve loop.
    for (iₜ, t) ∈ enumerate(Δt:Δt:T)
        ## First we solve the heat problem
        uₜ .= Alu \ (M * uₜ₋₁)

        ## Then we solve the point-wise reaction problem with the solution of
        ## the heat problem as initial guess.
        rvₜ = reshape(uₜ, (2, length(grid.nodes)))
        for i ∈ 1:length(grid.nodes)
            r₁ = rvₜ[1, i]
            r₂ = rvₜ[2, i]
            rvₜ[1, i] += Δt*( -r₁*r₂^2 + F *(1 - r₁) )
            rvₜ[2, i] += Δt*(  r₁*r₂^2 - r₂*(F + k ) )
        end

        ## The solution is then stored every 10th step to vtk files for
        ## later visualization purposes.
        if (iₜ % 10) == 0
            vtk_grid("reactive-surface-$t.vtu", dh) do vtk
                vtk_point_data(vtk, dh, uₜ)
                vtk_save(vtk)
                pvd[t] = vtk
            end
        end

        ## Finally we totate the solution to initialize the next timestep.
        uₜ₋₁ .= uₜ
    end

    vtk_save(pvd);
end

## This parametrization gives the spot pattern shown in the gif above.
if false #src
gray_scott_sphere(0.06, 0.062, 10.0, 32000.0)
else #src
gray_scott_sphere(0.06, 0.062, 10.0, 20.0) #src
end #src

#md # ## [Plain program](@id reactive_surface-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`reactive_surface.jl`](reactive_surface.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
