"""
ncurve_whichd,ncurve_space_fill_curve_numbering = ncurve_python_subdomain_space_filling_curve(cola,fina,iuse_starting_node,graph_trim,ncurve,[nonods,ncola])

Wrapper for ``ncurve_python_subdomain_space_filling_curve``.

Parameters
----------
cola : input rank-1 array('i') with bounds (ncola)
fina : input rank-1 array('i') with bounds (1 + nonods)
iuse_starting_node : input int
graph_trim : input int
ncurve : input int

Other Parameters
----------------
nonods : input int, optional
    Default: -1 + shape(fina, 0)
ncola : input int, optional
    Default: shape(cola, 0)

Returns
-------
ncurve_whichd : rank-2 array('i') with bounds (nonods,ncurve)
ncurve_space_fill_curve_numbering : rank-2 array('i') with bounds (nonods,ncurve)
"""

"""
a_sfc_all_un,fina_sfc_all_un,cola_sfc_all_un,ncola_sfc_all_un,b_sfc,
    ml_sfc,fin_sfc_nonods,nonods_sfc_all_grids,nlevel = 
    best_sfc_mapping_to_sfc_matrix_unstructured(a,b,ml,fina,cola,
    sfc_node_ordering,max_nonods_sfc_all_grids,max_ncola_sfc_all_un,max_nlevel,[ncola,nonods])

Wrapper for ``best_sfc_mapping_to_sfc_matrix_unstructured``.

Parameters
----------
a : input rank-1 array('f') with bounds (ncola)
b : input rank-1 array('f') with bounds (nonods)
ml : input rank-1 array('f') with bounds (nonods)
fina : input rank-1 array('i') with bounds (1 + nonods)
cola : input rank-1 array('i') with bounds (ncola)
sfc_node_ordering : input rank-1 array('i') with bounds (nonods)
max_nonods_sfc_all_grids : input int
max_ncola_sfc_all_un : input int
max_nlevel : input int

Other Parameters
----------------
ncola : input int, optional
    Default: shape(a, 0)
nonods : input int, optional
    Default: shape(b, 0)

Returns
-------
a_sfc_all_un : rank-1 array('f') with bounds (max_ncola_sfc_all_un)
fina_sfc_all_un : rank-1 array('i') with bounds (1 + max_nonods_sfc_all_grids)
cola_sfc_all_un : rank-1 array('i') with bounds (max_ncola_sfc_all_un)
ncola_sfc_all_un : int
b_sfc : rank-1 array('f') with bounds (max_nonods_sfc_all_grids)
ml_sfc : rank-1 array('f') with bounds (max_nonods_sfc_all_grids)
fin_sfc_nonods : rank-1 array('i') with bounds (1 + max_nlevel)
nonods_sfc_all_grids : int
nlevel : int

"""

"""
ncolele,finele,colele,midele = getfinele(totele,nloc,snloc,nonods,ndglno,mx_nface_p1,mxnele)

Wrapper for ``getfinele``.

Parameters
----------
totele : input int
nloc : input int
snloc : input int
nonods : input int
ndglno : input rank-1 array('i') with bounds (nloc * totele)
mx_nface_p1 : input int
mxnele : input int

Returns
-------
ncolele : int
finele : rank-1 array('i') with bounds (1 + totele)
colele : rank-1 array('i') with bounds (mxnele)
midele : rank-1 array('i') with bounds (totele)

"""

"""
values,indices,nidx,b_bc = classicip(sn,snx,sdetwei,snormal,nbele,nbf,c_bc,mx_nidx,eta_e,[nloc,nele,nface,sngi,ndim,nonods,glbnface])

Wrapper for ``classicip``.

Parameters
----------
sn : input rank-3 array('d') with bounds (nface,nloc,sngi)
snx : input rank-5 array('d') with bounds (nele,nface,ndim,nloc,sngi)
sdetwei : input rank-3 array('d') with bounds (nele,nface,sngi)
snormal : input rank-3 array('d') with bounds (nele,nface,ndim)
nbele : input rank-1 array('d') with bounds (glbnface)
nbf : input rank-1 array('d') with bounds (glbnface)
c_bc : input rank-1 array('d') with bounds (nonods)
mx_nidx : input int
eta_e : input float

Other Parameters
----------------
nloc : input int, optional
    Default: shape(sn, 1)
nele : input int, optional
    Default: shape(snx, 0)
nface : input int, optional
    Default: shape(sn, 0)
sngi : input int, optional
    Default: shape(sn, 2)
ndim : input int, optional
    Default: shape(snx, 2)
nonods : input int, optional
    Default: shape(c_bc, 0)
glbnface : input int, optional
    Default: shape(nbele, 0)

Returns
-------
values : rank-1 array('d') with bounds (mx_nidx)
indices : rank-2 array('i') with bounds (mx_nidx,2)
nidx : int
b_bc : rank-1 array('d') with bounds (nonods)

"""

"""
vec_a_sfc_all_un,
fina_sfc_all_un,
cola_sfc_all_un,
ncola_sfc_all_un,
vec_b_sfc,
ml_sfc,
fin_sfc_nonods,
nonods_sfc_all_grids,
nlevel = vector_best_sfc_mapping_to_sfc_matrix_unstructured(
    vec_a,
    vec_b,
    ml,
    fina,
    cola,
    sfc_node_ordering,
    max_nonods_sfc_all_grids,
    max_ncola_sfc_all_un,
    max_nlevel,
    [ndim,ncola,nonods])

Wrapper for ``vector_best_sfc_mapping_to_sfc_matrix_unstructured``.

Parameters
----------
vec_a : input rank-3 array('f') with bounds (ndim,ndim,ncola)
vec_b : input rank-2 array('f') with bounds (ndim,nonods)
ml : input rank-1 array('f') with bounds (nonods)
fina : input rank-1 array('i') with bounds (1 + nonods)
cola : input rank-1 array('i') with bounds (ncola)
sfc_node_ordering : input rank-1 array('i') with bounds (nonods)
max_nonods_sfc_all_grids : input int
max_ncola_sfc_all_un : input int
max_nlevel : input int

Other Parameters
----------------
ndim : input int, optional
    Default: shape(vec_a, 0)
ncola : input int, optional
    Default: shape(vec_a, 2)
nonods : input int, optional
    Default: shape(vec_b, 1)

Returns
-------
vec_a_sfc_all_un : rank-3 array('f') with bounds (ndim,ndim,max_ncola_sfc_all_un)
fina_sfc_all_un : rank-1 array('i') with bounds (1 + max_nonods_sfc_all_grids)
cola_sfc_all_un : rank-1 array('i') with bounds (max_ncola_sfc_all_un)
ncola_sfc_all_un : int
vec_b_sfc : rank-2 array('f') with bounds (ndim,max_nonods_sfc_all_grids)
ml_sfc : rank-1 array('f') with bounds (max_nonods_sfc_all_grids)
fin_sfc_nonods : rank-1 array('i') with bounds (1 + max_nlevel)
nonods_sfc_all_grids : int
nlevel : int
"""


"""
values,indices,nidx,rhs = stokes_assemble_fortran(
    sn,snx,sdetwei,snormal,sq,detwei,
    u_bc,
    nbele,nbf,alnmt,gi_align,
    mx_nidx,eta_e,
    [u_nloc,p_nloc,nele,nface,sngi,ndim,ngi]
)

Wrapper for ``stokes_assemble_fortran``.

Parameters
----------
sn : input rank-3 array('d') with bounds (nface,u_nloc,sngi)
snx : input rank-5 array('d') with bounds (nele,nface,ndim,u_nloc,sngi)
sdetwei : input rank-3 array('d') with bounds (nele,nface,sngi)
snormal : input rank-3 array('d') with bounds (nele,nface,ndim)
sq : input rank-3 array('d') with bounds (nface,p_nloc,sngi)
detwei : input rank-2 array('d') with bounds (nele,ngi)
u_bc : input rank-3 array('d') with bounds (nele,u_nloc,ndim)
nbele : input rank-1 array('i') with bounds (nele * nface)
nbf : input rank-1 array('i') with bounds (nele * nface)
alnmt : input rank-1 array('i') with bounds (nele * nface)
gi_align : input rank-2 array('i') with bounds (-1 + nface,sngi)
mx_nidx : input int
eta_e : input float

Other Parameters
----------------
u_nloc : input int, optional
    Default: shape(sn, 1)
p_nloc : input int, optional
    Default: shape(sq, 1)
nele : input int, optional
    Default: shape(snx, 0)
nface : input int, optional
    Default: shape(sn, 0)
sngi : input int, optional
    Default: shape(sn, 2)
ndim : input int, optional
    Default: shape(snx, 2)
ngi : input int, optional
    Default: shape(detwei, 1)

Returns
-------
values : rank-1 array('d') with bounds (mx_nidx)
indices : rank-2 array('i') with bounds (mx_nidx,2)
nidx : int
rhs : rank-1 array('d') with bounds (ndim * nele * u_nloc + nele * p_nloc)
"""
