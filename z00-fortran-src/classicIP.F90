subroutine classicIP(sn, snx, sdetwei, snormal, nbele, nbf, c_bc, &
  nloc, nele, nface, sngi, ndim, nonods, glbnface, &
  mx_nidx, values, indices, nidx, b_bc, eta_e)
  ! this subroutine takes in face shape functions and calculate 
  ! surface integral matrix using classical interior penalty 
  ! method (c.f. Arnold 2001).
  ! output is in coo format, i.e. :
  ! values - values of entries
  ! indices - (i,j) coordinates of entris. May have repeated indices
  !   that is going to be automatically sum up
  ! nidx - length of values / indices
  ! b_bc - rhs due to boundary conditions

  implicit none 
  integer, intent(in) :: nloc, nele, nface, sngi, ndim, nonods, glbnface, mx_nidx
  real(kind=8), dimension(nface, nloc, sngi), intent(in) :: sn
  real(kind=8), dimension(nele, nface, ndim, nloc, sngi), intent(in) :: snx
  real(kind=8), dimension(nele, nface, sngi), intent(in) :: sdetwei
  real(kind=8), dimension(nele, nface, ndim), intent(in) :: snormal
  ! real(kind=8), dimension(nonods, ndim), intent(in) :: x_all
  real(kind=8), dimension(glbnface), intent(in) :: nbele, nbf ! python idx starts from 0
  real(kind=8), dimension(nonods), intent(in) :: c_bc ! diri bc
  real(kind=8), dimension(mx_nidx), intent(out) :: values 
  integer, dimension(mx_nidx,2), intent(out) :: indices
  integer, intent(out) :: nidx
  real(kind=8), dimension(nonods), intent(out) :: b_bc
  real(8), intent(in) :: eta_e ! penalty coeeff
  ! local variable
  real(8) :: mu_e , ele2f
  integer :: ele, ele2, iface, iface2, glb_iface, inod, glb_inod
  integer :: glb_jnod, sgi, sgi2, idim, jnod , glb_iface2
  integer :: jnod2, glb_jnod2 , iii
  real(8) :: nnx, nxn, nn 

  iii = 1
  b_bc = 0.
  ! print*,nbf
  ! print*,'nface, nele',nface, nele
  do ele = 1,nele 
    do iface = 1,nface 
      mu_e = eta_e / sum(sdetwei(ele,iface,:))
      glb_iface = (ele-1)*3+iface 
      ele2f = nbele(glb_iface)
      if (isnan(ele2f)) then
        ! this is a boundary face
        ! print*, 'ele ', ele, 'ele2 ','nan   ','iface ',iface,'iface2 ','nan '
        do inod = 1,nloc 
          ! this side
          glb_inod = (ele-1)*nloc + inod 
          do jnod = 1,nloc
            glb_jnod = (ele-1)*nloc + jnod 
            nnx = 0.
            nxn = 0.
            nn = 0.
            do idim = 1,ndim 
              nnx = nnx + sum(sn(iface,jnod,:)*snx(ele,iface,idim,inod,:)*sdetwei(ele,iface,:)) &
                * snormal(ele,iface,idim) ! Nj1 * Ni1x * n1
              nxn = nxn + sum(snx(ele,iface,idim,jnod,:)*sn(iface,inod,:)*sdetwei(ele,iface,:)) &
                * snormal(ele,iface,idim) ! Nj1x * Ni1 * n1
            enddo !idim
            nn = nn + sum(sn(iface,jnod,:)*sn(iface,inod,:)*sdetwei(ele,iface,:)) ! Nj1n1 * Ni1n1 ! n1 \cdot n1 = 1
            ! print *, 'glbi, ',glb_inod, 'glbj, ', glb_jnod, 'nnx ',nnx,'nxn ',nxn,'nn ',nn
            ! sum to values and record the index
            indices(iii,1) = glb_inod
            indices(iii,2) = glb_jnod
            values(iii) = -nnx-nxn+mu_e*nn 
            iii = iii+1
            ! compute rhs
            b_bc(glb_inod) = b_bc(glb_inod) + c_bc(glb_jnod) * (-nnx+mu_e*nn) 
          enddo !jnod
        enddo !inod
        cycle
      endif !isnan(ele2f)
      ele2 = int(abs(ele2f))+1 ! cast to integer index
      glb_iface2 = int(abs(nbf(glb_iface))) + 1 ! from python idx to fortran idx
      iface2 = mod( glb_iface2-1, 3 )+1
      ! print*, 'ele ', ele, 'ele2f ', ele2f, 'ele2 ',ele2,'iface ',iface,'iface2',iface2
      do inod = 1,nloc 
        glb_inod = (ele-1)*nloc+inod 
        ! this side 
        do jnod = 1,nloc 
          glb_jnod = (ele-1)*nloc + jnod 
          nnx = 0.
          nxn = 0.
          nn = 0.
          do idim = 1,ndim 
            nnx = nnx + sum( sn(iface,jnod,:)*snx(ele,iface,idim,inod,:)*sdetwei(ele,iface,:) ) &
              * snormal(ele,iface,idim)
            nxn = nxn + sum( snx(ele,iface,idim,jnod,:)*sn(iface,inod,:)*sdetwei(ele,iface,:) ) &
              * snormal(ele,iface,idim)
          enddo !idim
          nn = nn + sum( sn(iface,jnod,:)*sn(iface,inod,:)*sdetwei(ele,iface,:) )
          ! print *, 'glbi, ',glb_inod, 'glbj, ', glb_jnod, 'nnx ',nnx,'nxn ',nxn,'nn ',nn
          ! sum and put index to indices
          indices(iii,1) = glb_inod
          indices(iii,2) = glb_jnod 
          values(iii) = -0.5*nnx -0.5*nxn +mu_e*nn
          iii = iii+1
        enddo !jnod
        ! other side
        do jnod2 = 1,nloc 
          glb_jnod2 = (ele2-1)*nloc + jnod2 
          nnx = 0.
          nxn = 0.
          nn = 0.
          do sgi = 1,sngi 
            sgi2 = sngi+1 - sgi ! sgi on the other side
            do idim = 1,ndim 
              nnx = nnx + sn(iface2,jnod2,sgi2)*snx(ele,iface,idim,inod,sgi)*sdetwei(ele,iface,sgi) &
                * snormal(ele2,iface2,idim)
              nxn = nxn + snx(ele2,iface2,idim,jnod2,sgi2)*sn(iface,inod,sgi)*sdetwei(ele,iface,sgi) &
                * snormal(ele,iface,idim)
            enddo !idim
            nn = nn + (-1.)*sn(iface2,jnod2,sgi2)*sn(iface,inod,sgi)*sdetwei(ele,iface,sgi)
          enddo !sgi
          ! print *, 'glbi, ',glb_inod, 'glbj, ', glb_jnod2, 'nnx ',nnx,'nxn ',nxn,'nn ',nn
          ! sum and put index to indices
          indices(iii,1) = glb_inod 
          indices(iii,2) = glb_jnod2
          values(iii) = -0.5*nnx-0.5*nxn+mu_e*nn 
          iii = iii+1 
        enddo !jnod2
      enddo !inod
    enddo !iface
  enddo !ele
  nidx = iii-1
end subroutine 

! integer function sgi2(sgi)
!   ! return gaussian pnts index on the other side
!   ! in fortran idx (starting from 1)
!   implicit none 
!   integer, intent(in) :: sgi
!   integer, dimension(4) :: order_on_other_side = [4,3,2,1]
!   sgi2 = order_on_other_side(sgi)
! end function sgi2