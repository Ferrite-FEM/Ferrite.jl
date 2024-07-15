# Yu, Jinyun. Symmetric Gaussian Quadrature Formulae for Tetrahedronal Regions. 1984. CMAME.
function _get_jinyun_tet_quadrature_data(n::Int)
    if n == 1
        a = 1. / 4.
        w = 1. / 6.
        xw = [a a a w]
    elseif n == 2
        a = ( 5. + 3. * √(5.) ) / 20.
        b = ( 5. - √(5.) ) / 20.
        w = 1. / 24.
        xw = [a b b w
              b a b w
              b b a w
              b b b w]
    elseif n == 3
        a1 = 1. / 4.
        a2 = 1. / 2.
        b2 = 1. / 6.
        w1 = -2. / 15.
        w2 = 3. / 40.
        xw = [a1 a1 a1 w1
              a2 b2 b2 w2
              b2 a2 b2 w2
              b2 b2 a2 w2
              b2 b2 b2 w2]
    elseif 4 ≤ n ≤ 6
        throw(ArgumentError("Jinyun's Gauss quadrature rule (RefTetrahedron) is not implemented for orders 4 and 6"))
    else
        throw(ArgumentError("unsupported quadrature order $n for Jinyun's Gauss quadrature rule (RefTetrahedron). Supported orders are 1 to 3."))
    end
    return xw
end

# Patrick Keast. Moderate-Degree Tetrahedral Quadrature Formulas. 1986. CMAME.
# Minimal points
function _get_keast_a_tet_quadrature_data(n::Int)
    if 1 ≤ n ≤ 3
        # The rules of Jinyin and Keast are identical for order 1 to 3, as stated in the Keast paper.
        xw = _get_jinyun_tet_quadrature_data(n)
    elseif n == 4
        a1 = 1. / 4.;
        w1 = -74. / 5625.;

        a2 = 5. / 70.;
        b2 = 11. / 14.;
        w2 = 343. / 45000.;

        a3 = ( 1. + √(5. / 14.) ) / 4.;
        b3 = ( 1. - √(5. / 14.) ) / 4.;
        w3 = 28. / 1125.;

        xw = [a1 a1 a1 w1
              b2 a2 a2 w2
              a2 b2 a2 w2
              a2 a2 b2 w2
              a2 a2 a2 w2
              a3 a3 b3 w3
              a3 b3 a3 w3
              a3 b3 b3 w3
              b3 a3 a3 w3
              b3 a3 b3 w3
              b3 b3 a3 w3]
    elseif n == 5
        w1 = 0.602678571428571597e-2
        a1 = 1. / 3.
        b1 = 0.

        w2 = 0.302836780970891856e-1
        a2 = 1. / 4.

        w3 = 0.116452490860289742e-1
        a3 = 1. / 11.
        b3 = 8. / 11.

        w4 = 0.109491415613864534e-1
        a4 = 0.665501535736642813e-1
        b4 = 0.433449846426335728e-0

        xw = [a1 a1 a1 w1
              a1 a1 b1 w1
              a1 b1 a1 w1
              b1 a1 a1 w1
              a2 a2 a2 w2
              a3 a3 a3 w3
              a3 a3 b3 w3
              a3 b3 a3 w3
              b3 a3 a3 w3
              a4 a4 b4 w4
              a4 b4 a4 w4
              a4 b4 b4 w4
              b4 a4 a4 w4
              b4 a4 b4 w4
              b4 b4 a4 w4]
    elseif 6 ≤ n ≤ 8
        throw(ArgumentError("Keast's Gauss quadrature rule (RefTetrahedron) not implement for order 6 to 8"))
    else
        throw(ArgumentError("unsupported order $n for Keast's Gauss quadrature rule (RefTetrahedron). Supported orders are 1 to 5."))
    end
    return xw
end

# Positive points
function _get_keast_b_tet_quadrature_data(n::Int)
    if n == 4
        w1 = 0.317460317460317450e-2
        a1 = 1. / 2.
        b1 = 0.0

        w2 = 0.147649707904967828e-1
        a2 = 0.100526765225204467e-0
        b2 = 0.698419704324386603e-0

        w3 = 0.221397911142651221e-1
        a3 = 0.314372873493192195e-0
        b3 = 0.568813795204234229e-1

        xw = [a1 a1 b1 w1
              a1 b1 a1 w1
              a1 b1 b1 w1
              b1 a1 a1 w1
              b1 a1 b1 w1
              b1 b1 a1 w1
              a2 a2 a2 w2
              a2 a2 b2 w2
              a2 b2 a2 w2
              b2 a2 a2 w2
              a3 a3 a3 w3
              a3 a3 b3 w3
              a3 b3 a3 w3
              b3 a3 a3 w3]
    else
        xw = _get_keast_a_tet_quadrature_data(n)
    end
    return xw
end
