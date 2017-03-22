# Patrick Keast, MODERATE-DEGREE TETRAHEDRAL QUADRATURE FORMULAS
function _get_gauss_tetdata(n::Int)
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
    else
        throw(ArgumentError("unsupported order for tetraheder gauss-legendre integration"))
    end
    return xw
end
