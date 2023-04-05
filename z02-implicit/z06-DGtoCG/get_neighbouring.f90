subroutine getfinele( totele, nloc, snloc, nonods, ndglno, mx_nface_p1, &
    mxnele, ncolele, finele, colele, midele )
    ! This sub caluculates COLELE the element connectivitiy list
    ! in order of faces.
    implicit none
    integer, intent( in ) :: totele, nloc, snloc, nonods
    integer, dimension( totele * nloc ), intent( in ) :: ndglno
    integer, intent( in ) :: mx_nface_p1, mxnele
    integer, intent( out ) :: ncolele
    integer, dimension( mxnele ), intent( out ) :: colele
    integer, dimension( totele + 1 ), intent( out ) :: finele
    integer, dimension( totele ), intent( out ) :: midele
    ! Local variables
    integer :: ele, iloc, jloc, iloc2, nod, inod, jnod, count, ele2, i, hit, &
        iface, itemp, count2
    logical :: found
    integer, allocatable, dimension( : ) :: fintran, coltran, icount

    allocate( fintran( nonods + 1 ))
    allocate( coltran( max( totele, nonods ) * mx_nface_p1 ))
    allocate( icount( max( nonods, totele) ))

    icount = 0
    do ele = 1, totele
        do iloc = 1, nloc
            nod = ndglno( ( ele - 1 ) * nloc + iloc )
            icount( nod ) = icount( nod ) + 1
        end do
    end do

    fintran = 0
    fintran( 1 ) = 1
    do nod = 1, nonods
        fintran( nod + 1 ) = fintran( nod ) + icount( nod )
    end do

    icount = 0
    coltran = 0
    do ele = 1, totele
        do iloc = 1, nloc
            nod = ndglno( ( ele - 1 ) * nloc + iloc )
            !ewrite(3,*)'nod, filtran, icount, ele:', nod, fintran( nod ), ele, totele, nonods
            coltran( fintran( nod ) + icount( nod )) = ele
            icount( nod ) = icount( nod ) + 1
        end do
    end do
    !ewrite(3,*)'coltran:', coltran( 1: max(totele, nonods ) * mx_nface_p1 )
    !ewrite(3,*)'fintran:', fintran( 1: nonods + 1 )
    !ewrite(3,*)'X_NDGLN:', ndglno( 1: totele*nloc )

    icount = 0 ; colele = 0 ; ncolele = 0
    Loop_Elements1: do ele = 1, totele

        Loop_Iloc: do iloc = 1, nloc

            nod = ndglno( ( ele - 1 ) * nloc + iloc )
            Loop_Count1: do count = fintran( nod ), fintran( nod + 1 ) - 1, 1

                ele2 = coltran( count )
                found = .false. ! Add ELE2 into list FINELE and COLELE
                do i = 1, icount( ele )
                    if( colele( ( ele - 1 ) * mx_nface_p1 + i ) == ele2 ) found = .true.
                end do

                Conditional_Found: if ( .not. found ) then ! Do elements ELE and ELE2 share at least 3 nodes?

                    hit = 0
                    do iloc2 = 1, nloc
                        inod = ndglno( ( ele - 1 ) * nloc + iloc2 )
                        do jloc = 1, nloc
                            jnod = ndglno( ( ele2 - 1 ) * nloc + jloc )
                            if ( inod == jnod ) hit = hit + 1
                        end do
                    end do
                    if ( hit >= snloc ) then
                        icount( ele ) = icount( ele ) + 1
                        colele( ( ele - 1 ) * mx_nface_p1 + icount( ele )) = ele2
                        ncolele = ncolele + 1
                    end if

                end if Conditional_Found

            end do Loop_Count1

        end do Loop_Iloc

    end do Loop_Elements1

    finele( 1 ) = 1
    do ele = 1, totele
        finele( ele + 1 ) = finele( ele ) + icount( ele )
    end do

    ! order elements in increasing order...
    count = 0
    Loop_Elements2: do ele = 1, totele
        ! Shorten COLELE then perform a bubble sort to get the ordering right for.
        do iface = 1, mx_nface_p1
            if ( colele( ( ele - 1 ) * mx_nface_p1 + iface ) /= 0 ) then
                count = count + 1
                colele( count ) = colele( ( ele - 1 ) * mx_nface_p1 + iface )
            end if
        end do
    end do Loop_Elements2

    Loop_BubbleSort: do ele = 1, totele
        do count = finele( ele ) , finele( ele + 1 ) - 2
            do count2 = finele( ele ) , finele( ele + 1 ) - 1, 1
                if ( colele( count ) > colele( count + 1 )) then ! swop over
                    itemp = colele( count + 1 )
                    colele( count + 1 ) = colele( count )
                    colele( count ) = itemp
                end if
            end do
        end do
    end do Loop_BubbleSort

    ! Calculate midele:
    do ele = 1, totele
        do count = finele( ele ) , finele( ele + 1 ) - 1
            if(colele(count)==ele) midele( ele ) = count
        end do
    end do

    deallocate( fintran )
    deallocate( coltran )
    deallocate( icount )

    return
end subroutine getfinele


subroutine getfin_p1cg(cg_noglbn, nele, nloc, p1dg_nonods, idx, n_idx)
    implicit none
    integer, intent(in) :: nele, nloc, p1dg_nonods
    integer, intent(in) :: cg_noglbn(p1dg_nonods)
    integer, dimension(2, nele*9), intent(out) :: idx
    integer, intent(out) :: n_idx  ! length of idx
    ! local variables
    integer :: ele, inod, jnod, glbi, glbj

    n_idx = 0
    do ele = 1,nele
        do inod = 1,nloc
            glbi = cg_noglbn((ele-1)*nloc+inod)
            do jnod = 1,nloc
                glbj = cg_noglbn((ele-1)*nloc+jnod)
                n_idx = n_idx+1
                idx(1,n_idx) = glbi
                idx(2,n_idx) = glbj
            end do
        end do
    end do

    return
end subroutine getfin_p1cg