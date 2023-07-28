subroutine stokes_assemble_fortran(sn, snx, sdetwei, snormal, sq, detwei, &  ! shape functions
  u_bc, &  ! boundary condition
  nbele, nbf, alnmt, &  ! neighbour info
  u_nloc, p_nloc, nele, nface, sngi, ndim, ngi, &  ! constants
  gi_align, &  ! gi alignment on the other side
  mx_nidx, values, indices, nidx, rhs, &  ! output array and dimension info
  eta_e)  ! penalty coefficient

  implicit none 
  integer, intent(in) :: u_nloc, p_nloc, nele, nface, sngi, ndim, ngi
  real(kind=8), intent(in) :: sn(nface, u_nloc, sngi), snx(nele, nface, ndim, u_nloc, sngi), &
    sdetwei(nele, nface, sngi), snormal(nele, nface, ndim), &
    sq(nface, p_nloc, sngi), detwei(nele, ngi), &
    u_bc(nele, u_nloc, ndim)
  integer, intent(in) :: nbele(nface*nele), nbf(nface*nele), alnmt(nface*nele), &
    gi_align(nface-1, sngi)
  integer, intent(in) :: mx_nidx
  real(8), intent(in) :: eta_e
  real(kind=8), intent(out) :: values(mx_nidx), rhs(nele*ndim*u_nloc+nele*p_nloc)
  integer, intent(out) :: indices(mx_nidx, 2)
  integer, intent(out) :: nidx 
  ! local variables
  real(8) :: mu_e 
  integer :: ele, ele2, iface, iface2, glb_iface, inod, glb_inod, &
    glb_jnod, idim, jnod, glb_iface2, jnod2, glb_jnod2, iii, &
    jdim, kdim, jdim2
  real(8) :: vnux, vxun, nn, qun, vnp 

  iii = 1
  rhs = 0.
  values = 0.
  do ele = 1, nele
    do iface = 1, nface 
      glb_iface = (ele-1) * nface + iface 
      if (alnmt(glb_iface)<0) then 
        ! this is a boundary face
        mu_e = eta_e / (sum(detwei(ele, :)))**(1./ndim)
        ! K and G
        do inod = 1, u_nloc 
          do idim = 1, ndim 
            glb_inod = (ele-1) * u_nloc * ndim + (inod - 1) * ndim + idim
            ! K
            do jnod = 1, u_nloc 
              jdim = idim 
              glb_jnod = (ele-1) * u_nloc * ndim + (jnod - 1) * ndim + jdim
              vnux = 0
              vxun = 0
              nn = 0
              do kdim = 1, ndim
                vnux = vnux + sum(sn(iface, inod, :) * snormal(ele, iface, kdim) &
                                  * snx(ele, iface, kdim, jnod, :) &
                                  * sdetwei(ele, iface, :))
                vxun = vxun + sum(snx(ele, iface, kdim, inod, :) &
                                  * sn(iface, jnod, :) &
                                  * snormal(ele, iface, kdim) &
                                  * sdetwei(ele, iface, :))
              enddo
              nn = nn + sum(sn(iface, inod, :) * sn(iface, jnod, :) &
                            * sdetwei(ele, iface, :)) * mu_e
              indices(iii,1) = glb_inod 
              indices(iii,2) = glb_jnod
              values(iii) = -vnux -vxun + nn 
              iii = iii + 1
              ! bc contribution to rhs
              rhs(glb_inod) = rhs(glb_inod) + u_bc(ele, jnod, jdim) * (-vxun + nn)
            enddo  ! jnod
            ! G 
            do jnod = 1, p_nloc 
              glb_jnod = nele*u_nloc*ndim + (ele-1)*p_nloc + jnod 
              vnp = sum(sn(iface, inod, :) * snormal(ele, iface, idim) &
                        * sq(iface, jnod, :) * sdetwei(ele, iface, :))
              indices(iii, 1) = glb_inod 
              indices(iii, 2) = glb_jnod 
              values(iii) = vnp 
              iii = iii + 1
            enddo  ! jnod
          enddo  ! idim
        enddo  ! inod
        ! G^T
        do inod = 1, p_nloc 
          glb_inod = nele*u_nloc*ndim + (ele-1)*p_nloc + inod 
          do jnod = 1, u_nloc 
            do jdim = 1, ndim 
              glb_jnod = (ele-1)*u_nloc*ndim + (jnod-1)*ndim + jdim 
              qun = sum(sq(iface, inod, :) * sn(iface, jnod, :) &
                        * snormal(ele, iface, jdim) * sdetwei(ele, iface, :))
              indices(iii, 1) = glb_inod 
              indices(iii, 2) = glb_jnod 
              values(iii) = qun 
              iii = iii + 1
              ! add bc contribution to rhs
              rhs(glb_inod) = rhs(glb_inod) + u_bc(ele, jnod, jdim) * qun 
            enddo  ! jdim
          enddo  ! jnod
        enddo  ! inod
      else 
        ! this is interior face
        ele2 = nbele(glb_iface) + 1
        mu_e = 2. * eta_e / ( &
          (sum(detwei(ele, :))) ** (1./ndim) &
          + (sum(detwei(ele2, :))) ** (1./ndim) &
        )
        glb_iface2 = nbf(glb_iface) + 1
        iface2 = mod(glb_iface2, nface)
        ! K and G
        do inod = 1, u_nloc 
          do idim = 1, ndim 
            glb_inod = (ele-1) * u_nloc * ndim + (inod-1)*ndim + idim 
            ! this side 
            ! K
            do jnod = 1, u_nloc 
              jdim = idim 
              glb_jnod = (ele-1) * u_nloc * ndim + (jnod-1)*ndim + jdim 
              vnux = 0
              vxun = 0
              nn = 0
              do kdim = 1, ndim 
                vnux = vnux + sum(sn(iface, inod, :) &
                                  * snormal(ele, iface, kdim) &
                                  * snx(ele, iface, kdim, jnod, :) &
                                  * sdetwei(ele, iface, :))
                vxun = vxun + sum(snx(ele, iface, kdim, inod, :) &
                                  * sn(iface, jnod, :) &
                                  * snormal(ele, iface, kdim) &
                                  * sdetwei(ele, iface, :))
              enddo  ! kdim
              nn = nn + sum(sn(iface, inod, :) &
                            * sn(iface, jnod, :) &
                            * sdetwei(ele, iface, :)) * mu_e 
              indices(iii,1) = glb_inod 
              indices(iii,2) = glb_jnod 
              values(iii) = -0.5*vnux -0.5*vxun + nn
              iii = iii + 1
            enddo  ! jnod
            ! G
            do jnod = 1, p_nloc 
              glb_jnod = nele*u_nloc*ndim + (ele-1)*p_nloc + jnod
              vnp = sum(sn(iface,inod,:)*snormal(ele,iface,idim) &
                        * sq(iface, jnod, :)*sdetwei(ele,iface, :))
              indices(iii, 1) = glb_inod
              indices(iii, 2) = glb_jnod
              values(iii) = 0.5 * vnp
              iii = iii + 1
            enddo  ! jnod
            ! other side
            ! K
            do jnod2 = 1, u_nloc 
              jdim2 = idim 
              glb_jnod2 = (ele2-1) * u_nloc * ndim + (jnod2-1)*ndim + jdim2 
              vnux = 0
              vxun = 0
              nn = 0
              do kdim = 1, ndim 
                vnux = vnux + sum(sn(iface, inod, :) &
                                  * snormal(ele, iface, kdim) &
                                  * snx(ele2, iface2, kdim, jnod2, gi_align(alnmt(glb_iface2),:)) &
                                  * sdetwei(ele, iface, :))
                vxun = vxun + sum(snx(ele, iface, kdim, inod, :) &
                                  * sn(iface2, jnod2, gi_align(alnmt(glb_iface2),:)) &
                                  * snormal(ele2, iface2, kdim) &
                                  * sdetwei(ele, iface, :))
              enddo  ! kdim
              nn = nn + sum(sn(iface, inod, :) &
                            * sn(iface2, jnod2, gi_align(alnmt(glb_iface2),:)) &
                            * sdetwei(ele, iface, :)) * mu_e * (-1.)
              indices(iii,1) = glb_inod 
              indices(iii,2) = glb_jnod2 
              values(iii) = -0.5*vnux -0.5*vxun + nn
              iii = iii + 1
            enddo  ! jnod2
            ! G
            do jnod2 = 1, p_nloc 
              glb_jnod2 = nele*u_nloc*ndim + (ele2-1)*p_nloc + jnod2
              vnp = sum(sn(iface,inod,:)*snormal(ele,iface,idim) &
                        * sq(iface2, jnod2, gi_align(alnmt(glb_iface2),:)) &
                        * sdetwei(ele,iface, :))
              indices(iii, 1) = glb_inod
              indices(iii, 2) = glb_jnod2
              values(iii) = 0.5 * vnp
              iii = iii + 1
            enddo  ! jnod2
          enddo  ! idim
        enddo  ! inod
        ! G^T
        do inod = 1, p_nloc 
          glb_inod = nele*u_nloc*ndim + (ele-1)*p_nloc + inod
          ! this side
          do jnod = 1, u_nloc 
            do jdim = 1, ndim 
              glb_jnod = (ele-1)*u_nloc*ndim + (jnod-1)*ndim + jdim 
              qun = sum(sq(iface, inod, :) * sn(iface, jnod, :) &
                        * snormal(ele, iface, jdim) * sdetwei(ele, iface, :))
              indices(iii, 1) = glb_inod 
              indices(iii, 2) = glb_jnod 
              values(iii) = qun * 0.5
              iii = iii + 1
            enddo  ! jdim
          enddo  ! jnod
          ! other side
          do jnod2 = 1, u_nloc 
            do jdim2 = 1, ndim 
              glb_jnod2 = (ele2-1)*u_nloc*ndim + (jnod2-1)*ndim + jdim2 
              qun = sum(sq(iface, inod, :) &
                        * sn(iface2, jnod2, gi_align(alnmt(glb_iface2),:)) &
                        * snormal(ele2, iface2, jdim2) * sdetwei(ele, iface, :))
              indices(iii, 1) = glb_inod 
              indices(iii, 2) = glb_jnod2 
              values(iii) = qun * 0.5
              iii = iii + 1
            enddo  ! jdim2
          enddo  ! jnod2
        enddo  ! inod
      endif  ! alnmt<0
    enddo
  enddo
  nidx = iii - 1
end subroutine
