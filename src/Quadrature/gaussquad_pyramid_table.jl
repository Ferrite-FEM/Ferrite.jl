# Symmetric quadrature rules takes from
#   Witherden, Freddie D., and Peter E. Vincent. "On the identification of 
#   symmetric quadrature rules for finite element methods." Computers & 
#   Mathematics with Applications 69.10 (2015): 1232-1241.
function _get_gauss_pyramiddata_polyquad(n::Int)
  if n == 1
      xw = [0 0  -0.5  2.6666666666666666666666666666666666667]
  elseif n == 2
      xw = [
                                               0                                          0   0.21658207711955775339238838942231815011   0.60287280353093925911579186632475611728
        0.71892105581179616210276971993495767914                                          0  -0.70932703285428855378369530000365161136   0.51594846578393185188771870008547763735
                                               0   0.71892105581179616210276971993495767914  -0.70932703285428855378369530000365161136   0.51594846578393185188771870008547763735
       -0.71892105581179616210276971993495767914                                          0  -0.70932703285428855378369530000365161136   0.51594846578393185188771870008547763735
                                               0  -0.71892105581179616210276971993495767914  -0.70932703285428855378369530000365161136   0.51594846578393185188771870008547763735
      ]
  elseif n == 3
      xw = [
                                              0                                          0   0.14285714077213670617734974746312582074   0.67254902379402809443607078852738472107
                                              0                                          0  -0.99999998864829993678698817507850804299   0.30000001617617323518941867705084375434
       0.56108361105873963414196154191891982155   0.56108361105873963414196154191891982155  -0.66666666666666666666666666666666666667   0.42352940667411633426029430027210954782
       0.56108361105873963414196154191891982155  -0.56108361105873963414196154191891982155  -0.66666666666666666666666666666666666667   0.42352940667411633426029430027210954782
      -0.56108361105873963414196154191891982155   0.56108361105873963414196154191891982155  -0.66666666666666666666666666666666666667   0.42352940667411633426029430027210954782
      -0.56108361105873963414196154191891982155  -0.56108361105873963414196154191891982155  -0.66666666666666666666666666666666666667   0.42352940667411633426029430027210954782
      ]
  elseif n == 4
      xw = [
                                               0                                          0   0.35446557777227471722730849524904581806   0.30331168845504517111391728481208001144
                                               0                                          0  -0.74972609378250711033655604388057044149   0.55168907357213937275730716433358729608
         0.6505815563982325146829577797417295398                                          0  -0.35523170084357268589593075201816127231   0.28353223437153468006819777082540613962
                                               0    0.6505815563982325146829577797417295398  -0.35523170084357268589593075201816127231   0.28353223437153468006819777082540613962
        -0.6505815563982325146829577797417295398                                          0  -0.35523170084357268589593075201816127231   0.28353223437153468006819777082540613962
                                               0   -0.6505815563982325146829577797417295398  -0.35523170084357268589593075201816127231   0.28353223437153468006819777082540613962
        0.65796699712169008954533549931479427127   0.65796699712169008954533549931479427127  -0.92150343220236930457646242598412224897   0.16938424178833585063066278355484370017
        0.65796699712169008954533549931479427127  -0.65796699712169008954533549931479427127  -0.92150343220236930457646242598412224897   0.16938424178833585063066278355484370017
       -0.65796699712169008954533549931479427127   0.65796699712169008954533549931479427127  -0.92150343220236930457646242598412224897   0.16938424178833585063066278355484370017
       -0.65796699712169008954533549931479427127  -0.65796699712169008954533549931479427127  -0.92150343220236930457646242598412224897   0.16938424178833585063066278355484370017
      ]
  elseif n == 5
    xw = [
                                               0                                          0   0.45971576156501338586164265377920811314   0.18249431975770692138374895897213800931
                                               0                                          0  -0.39919795837246198593385139590712322914   0.45172563864726406056400285032640105704
                                               0                                          0  -0.99999998701645569241460017355590234925   0.15654542887619877154120304977336547704
        0.70652603154632457420722562974792066862                                          0                                      -0.75   0.20384344839498724639142514342645843799
                                               0   0.70652603154632457420722562974792066862                                      -0.75   0.20384344839498724639142514342645843799
       -0.70652603154632457420722562974792066862                                          0                                      -0.75   0.20384344839498724639142514342645843799
                                               0  -0.70652603154632457420722562974792066862                                      -0.75   0.20384344839498724639142514342645843799
        0.70511712277882760181079385797948261057   0.70511712277882760181079385797948261057  -0.87777618587595407108464357252416911085   0.10578907087905457654220386143818487109
        0.70511712277882760181079385797948261057  -0.70511712277882760181079385797948261057  -0.87777618587595407108464357252416911085   0.10578907087905457654220386143818487109
       -0.70511712277882760181079385797948261057   0.70511712277882760181079385797948261057  -0.87777618587595407108464357252416911085   0.10578907087905457654220386143818487109
       -0.70511712277882760181079385797948261057  -0.70511712277882760181079385797948261057  -0.87777618587595407108464357252416911085   0.10578907087905457654220386143818487109
        0.43288286410354097685000790909815143591   0.43288286410354097685000790909815143591  -0.15279732576055038842025517341026975071   0.15934280057233240536079894703404722173
        0.43288286410354097685000790909815143591  -0.43288286410354097685000790909815143591  -0.15279732576055038842025517341026975071   0.15934280057233240536079894703404722173
       -0.43288286410354097685000790909815143591   0.43288286410354097685000790909815143591  -0.15279732576055038842025517341026975071   0.15934280057233240536079894703404722173
       -0.43288286410354097685000790909815143591  -0.43288286410354097685000790909815143591  -0.15279732576055038842025517341026975071   0.15934280057233240536079894703404722173
    ]
  elseif n==6
    xw = [
                                               0                                          0   0.61529159538791884731954955103613269638  0.067901049767045421348323914977088073991
                                               0                                          0  -0.99647238229436073479434618269377000101  0.042809368467176804517723726702995063594
                                               0                                          0  -0.72347438707253884945296925394581802655   0.31887881187339684620076177026181758884
                                               0                                          0  -0.15715217612872593659809486760257994179   0.27482845386440493216991697942238502071
        0.83459535111470834854355609376418165506                                          0  -0.80510531794907605316224881154681003606   0.09853490116354529166000184266852321827
                                               0   0.83459535111470834854355609376418165506  -0.80510531794907605316224881154681003606   0.09853490116354529166000184266852321827
       -0.83459535111470834854355609376418165506                                          0  -0.80510531794907605316224881154681003606   0.09853490116354529166000184266852321827
                                               0  -0.83459535111470834854355609376418165506  -0.80510531794907605316224881154681003606   0.09853490116354529166000184266852321827
         0.4339254093766990953588290760534534212                                          0   0.13214918124660180928234184789309315759  0.084233545301795354894729459206933067324
                                               0    0.4339254093766990953588290760534534212   0.13214918124660180928234184789309315759  0.084233545301795354894729459206933067324
        -0.4339254093766990953588290760534534212                                          0   0.13214918124660180928234184789309315759  0.084233545301795354894729459206933067324
                                               0   -0.4339254093766990953588290760534534212   0.13214918124660180928234184789309315759  0.084233545301795354894729459206933067324
          0.565680854425675497672813848515505886     0.565680854425675497672813848515505886  -0.94104453830855858213371584535608020131  0.099200345038528745418340651788127467643
          0.565680854425675497672813848515505886    -0.565680854425675497672813848515505886  -0.94104453830855858213371584535608020131  0.099200345038528745418340651788127467643
         -0.565680854425675497672813848515505886     0.565680854425675497672813848515505886  -0.94104453830855858213371584535608020131  0.099200345038528745418340651788127467643
         -0.565680854425675497672813848515505886    -0.565680854425675497672813848515505886  -0.94104453830855858213371584535608020131  0.099200345038528745418340651788127467643
        0.49807909178070594245642503041316019049   0.49807909178070594245642503041316019049  -0.47016827357574098280216051103358470857   0.19701969247180494546144986676345270691
        0.49807909178070594245642503041316019049  -0.49807909178070594245642503041316019049  -0.47016827357574098280216051103358470857   0.19701969247180494546144986676345270691
       -0.49807909178070594245642503041316019049   0.49807909178070594245642503041316019049  -0.47016827357574098280216051103358470857   0.19701969247180494546144986676345270691
       -0.49807909178070594245642503041316019049  -0.49807909178070594245642503041316019049  -0.47016827357574098280216051103358470857   0.19701969247180494546144986676345270691
        0.95089948721448243656573347456644144515   0.95089948721448243656573347456644144515  -0.90350185873612803037779492533363294921  0.011573761697986328172963248398558769736
        0.95089948721448243656573347456644144515  -0.95089948721448243656573347456644144515  -0.90350185873612803037779492533363294921  0.011573761697986328172963248398558769736
       -0.95089948721448243656573347456644144515   0.95089948721448243656573347456644144515  -0.90350185873612803037779492533363294921  0.011573761697986328172963248398558769736
       -0.95089948721448243656573347456644144515  -0.95089948721448243656573347456644144515  -0.90350185873612803037779492533363294921  0.011573761697986328172963248398558769736
    ]
  else
      throw(ArgumentError("unsupported order for prism polyquad integration"))
  end
    # Transform from [-1,1] × [-1,1] × [-1,1] pyramid with volume 8/3 and with 5th node in center
    # to pyramid [0,1] × [0,1] × [0,1] with volume 1/3 and with 5th node in corner
    f1 = (x,y,z) -> (0.5(x+1.0), 0.5(y+1.0), 0.5(z+1.0))
    f2 = (x,y,z) -> (2x-z, 2y-z, z)

    for i in axes(xw, 1)
        x,y,z,w = xw[i,:]
        x,y,z = f1(x,y,z)
        x,y,z = f2(x,y,z)

        xw[i, 1] = x
        xw[i, 2] = y
        xw[i, 3] = z
        xw[i, 4] = w * ((1/3)/(8/3))
    end

  return xw
end