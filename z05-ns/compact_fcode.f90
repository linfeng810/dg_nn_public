
    ! in python:
    ! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = best_sfc_mapping_to_sfc_matrix( &
    !                                     a, b, ml, &
    !                                     fina,cola, ncola, sfc_node_ordering, &
    !                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
    subroutine best_sfc_mapping_to_sfc_matrix_unstructured( &
        a_sfc_all_un,fina_sfc_all_un, &
        cola_sfc_all_un,ncola_sfc_all_un, &
        b_sfc, ml_sfc, &
        fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
        a, b, ml,  &
        fina,cola, sfc_node_ordering, ncola, &
        nonods, max_nonods_sfc_all_grids, &
        max_ncola_sfc_all_un, max_nlevel)  
        ! It does this with a kernal size of nfilt_size_sfc. 
        ! this subroutine finds the space filling curve representation of matrix eqns A T=b 
        ! - that is it forms matrix a and vector b and the soln vector is T 
        ! although T is not needed here. 
        ! It does this with a kernal size of nfilt_size_sfc. 
        ! It uses the BEST approach we can to form these tridigonal matrix approximations on different grids. 
        ! It also puts the vector b in space filling curve ordering. 
        ! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
        ! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
        ! which are stored in b_sfc, ml_sfc. 
        use, intrinsic :: iso_fortran_env
        implicit none
        ! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
        ! are nlevel grids from course to fine. 
        ! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
        ! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
        ! sfc_node_ordering(fem node number)=i_sfc_order. Here i_sfc_order is the number of the node meansured along 
        ! the space filling curve trajectory. 
        ! nonods=number of finite element nodes in the mesh.
        ! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
        ! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
        ! call in python: nlevel = calculate_nlevel_sfc(nonods). 
        !        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
        ! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
        !        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
        !        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
        ! 
        ! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
        ! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
        ! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
        ! fina(inod) start of the inod row of a matrix.
        ! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
        !      1-----2-----3
        !      !     !     !
        !      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
        !      4-----5-----6
        ! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
        !           1  2  3  4  5  6 - column
        ! row 1    (X  X  0  X  X  0)
        ! row 2    (X  X  X  X  X  X)
        ! row 3    (0  X  X  0  X  X)
        ! row 4    (X  X  0  X  X  0)
        ! row 5    (X  X  X  X  X  X)
        ! row 6    (0  X  X  0  X  X)
        ! The comparact row storage only stores the non-zeros. 
        ! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
        ! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
        ! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
        ! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
        ! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
        ! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
        !                                                                                             fina(7)=29
        ! 
        ! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids
        integer, intent( in ) :: max_ncola_sfc_all_un, max_nlevel
        real(real64), intent( out ) :: a_sfc_all_un(max_ncola_sfc_all_un), &
        b_sfc(max_nonods_sfc_all_grids), &
        ml_sfc(max_nonods_sfc_all_grids) 
        real(real64), intent( in ) :: a(ncola), b(nonods), ml(nonods)
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel+1), nlevel
        integer, intent( out ) :: fina_sfc_all_un(max_nonods_sfc_all_grids+1), &
        cola_sfc_all_un(max_ncola_sfc_all_un),ncola_sfc_all_un
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
        ! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:), in_row_count(:)
        integer, allocatable :: fina_sfc_all_un2(:)
        integer i, count, count2, nodj, nodi_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_all
        integer ifinest_nod, jfinest_nod, ipt
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer nrow, count_col, idisplace, jcourse_nod_sfc_all2
        integer jcourse_nod_sfc_all, ifinest_nod_sfc_all, count_all
        logical found
        ! 
        ! print *,'2-just inside best_sfc_mapping_to_sfc_matrix_unstructured'
        ! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
        ! 
        ! form SFC matrix...
        !        a_sfc(:,:)=0.0
        b_sfc(:)=0.0
        ml_sfc(:)=0.0
        do ifinest_nod=1,nonods
            ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
            b_sfc(ifinest_nod_sfc)=b(ifinest_nod)
            ml_sfc(ifinest_nod_sfc)=ml(ifinest_nod)
        end do 
        !       print *,'here 1 nlevel:',nlevel
        ! 
        ! coarsen...
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
        do ilevel=2,nlevel
            !           print *,'ilevel=',ilevel
            sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
            if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
            sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
            !           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
            !                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
            !           print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1):', &
            !                    fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1)
            !           print *,'max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine:', &
            !                    max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine
            
            call map_sfc_fine_grid_2_course_grid_vec( &
            ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
            ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
            !           ipt=fin_sfc_nonods(ilevel)
            !           print *,'ml_sfc(ipt:ipt+sfc_nonods_course-1):',ml_sfc(ipt:ipt+sfc_nonods_course-1)
            !           ipt=fin_sfc_nonods(ilevel-1)
            !           print *,'ml_sfc(ipt:ipt+sfc_nonods_fine-1):',ml_sfc(ipt:ipt+sfc_nonods_fine-1)
            !       print *,'here 1.1' 
            call map_sfc_fine_grid_2_course_grid_vec( &
            b_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
            b_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
            !       print *,'here 1.2' 
            sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
            fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum-1
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
            print *,'run out of memory here stopping'
            stop 2822
        endif
        ! print *,'here 2'
        !         stop 25
        ! sfc_node_ordering(nod) = new node numbering from current node number nod.
        allocate(sfc_node_ordering_inverse(nonods))
        do ifinest_nod=1,nonods
            ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
            sfc_node_ordering_inverse(ifinest_nod_sfc) = ifinest_nod
        end do 
        !     
        ! print *,'---max_ncola_sfc_all_un,ncola,nlevel:',max_ncola_sfc_all_un,ncola,nlevel
        ! print *,'---nonods,nonods_sfc_all_grids:',nonods,nonods_sfc_all_grids
        ! print *,'fin_sfc_nonods(1:,nlevel+1):',fin_sfc_nonods(1:nlevel+1)
        !        a_sfc_all_un=0.0
        count_all=0
        do ilevel=1,nlevel
            ilevel2=2**(ilevel-1)
            ! print *,'--- ilevel=',ilevel
            idisplace = fin_sfc_nonods(ilevel) 
            fina_sfc_all_un(idisplace) = count_all+1
            do ifinest_nod_sfc=1,nonods
                !              print *,'ifinest_nod_sfc=',ifinest_nod_sfc
                ifinest_nod = sfc_node_ordering_inverse(ifinest_nod_sfc)
                icourse_nod_sfc_all = idisplace + (ifinest_nod_sfc-1)/ilevel2
                do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                    jfinest_nod = cola(count)
                    jfinest_nod_sfc = sfc_node_ordering(jfinest_nod) 
                    jcourse_nod_sfc_all = idisplace + (jfinest_nod_sfc-1)/ilevel2
                    ! look to see if we have included jcourse_nod_sfc_all yet
                    found=.false.
                    do count2=fina_sfc_all_un(icourse_nod_sfc_all),count_all
                        jcourse_nod_sfc_all2=cola_sfc_all_un(count2)
                        if(jcourse_nod_sfc_all==jcourse_nod_sfc_all2) then
                            found=.true.
                            a_sfc_all_un(count2)=a_sfc_all_un(count2)+a(count) ! map from original matrix
                        endif
                    end do
                    if(.not.found) then
                        count_all=count_all+1
                        cola_sfc_all_un(count_all) = jcourse_nod_sfc_all
                        !                    a_sfc_all_un(count_all)=a_sfc_all_un(count_all)+a(count) ! map from original matrix
                        a_sfc_all_un(count_all)=a(count) ! map from original matrix
                    endif
                end do ! do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                fina_sfc_all_un(icourse_nod_sfc_all+1) = count_all+1
            end do ! do ifinest_nod_sfc=1,nonods
            ! print *,'here2'
            ! print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1):', &
            ! fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)
            ! print *,'diff=',fin_sfc_nonods(ilevel+1)-fin_sfc_nonods(ilevel)
            ! print *,'fina_sfc_all_un(fin_sfc_nonods(ilevel)):',fina_sfc_all_un(fin_sfc_nonods(ilevel))
            ! print *,'fina_sfc_all_un(fin_sfc_nonods(ilevel+1))-1:',fina_sfc_all_un(fin_sfc_nonods(ilevel+1))-1
            ! print *,'difference:', &
            ! fina_sfc_all_un(fin_sfc_nonods(ilevel+1)) - fina_sfc_all_un(fin_sfc_nonods(ilevel))
            if(.false.) then
                do icourse_nod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1
                    print *,'icourse_nod_sfc_all=',icourse_nod_sfc_all
                    print *,'cola_sfc_all_un(count2):', &
                    (cola_sfc_all_un(count2),count2=fina_sfc_all_un(icourse_nod_sfc_all), &
                    fina_sfc_all_un(icourse_nod_sfc_all+1)-1) 
                    print *,'a_sfc_all_un(count2):', &
                    (a_sfc_all_un(count2),count2=fina_sfc_all_un(icourse_nod_sfc_all), &
                    fina_sfc_all_un(icourse_nod_sfc_all+1)-1) 
                end do
            endif
        end do ! do ilevel=1,nlevel
        
        ncola_sfc_all_un = fina_sfc_all_un(nonods_sfc_all_grids+1)-1
        if( max_ncola_sfc_all_un < ncola_sfc_all_un ) then
            print *,'run out of memory here stopping'
            stop 2825
        endif
        ! 
        ! print *,'ncola, ncola_sfc_all_un:',ncola, ncola_sfc_all_un
        ! print *,'nonods_sfc_all_grids:',nonods_sfc_all_grids
        ! print *,'fina_sfc_all_un(nonods_sfc_all_grids+1)-fina_sfc_all_un(nonods_sfc_all_grids):', &
        ! fina_sfc_all_un(nonods_sfc_all_grids+1)-fina_sfc_all_un(nonods_sfc_all_grids)
        
        !        stop 282
        ! print *,'just leaving best_sfc_mapping_to_sfc_matrix_unstructured'
        return 
    end subroutine best_sfc_mapping_to_sfc_matrix_unstructured

    ! in python:
    ! nlevel = calculate_nlevel_sfc(nonods)
    subroutine calculate_nlevel_sfc(nlevel,nonods)
        ! this subroutine calculates the number of multi-grid levels for a 
        ! space filling curve multi-grid or 1d multi-grid applied to nonods nodes. 
        implicit none
        integer, intent( in ) :: nonods
        integer, intent( out ) :: nlevel
        ! local variables...
        integer sfc_nonods_fine,sfc_nonods_course,ilevel
        ! coarsen...
        if(nonods==1) then
            nlevel=1 
        else
            sfc_nonods_course=nonods
            do ilevel=2,200
                sfc_nonods_fine=sfc_nonods_course
                sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
                nlevel=ilevel
                if(sfc_nonods_course==1) exit
            end do
        endif
        return
    end subroutine calculate_nlevel_sfc

    
    subroutine map_sfc_fine_grid_2_course_grid_vec( ml_sfc_course,sfc_nonods_course, &
        ml_sfc_fine,sfc_nonods_fine)
        use, intrinsic :: iso_fortran_env
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        real(real64), intent( out ) :: ml_sfc_course(sfc_nonods_course)
        real(real64), intent( in ) :: ml_sfc_fine(sfc_nonods_fine)
        ! local variables...
        integer i_short, i_long
        ! 
        ml_sfc_course(:)=0.0
        !        do i_short=1,sfc_nonods_course
        !           i_long=(i_short-1)*2 + 1
        !           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        
        !           i_long=min( (i_short-1)*2 + 2, sfc_nonods_fine)----miss this out---
        !           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        !        end do
        ! 
        do i_long=1,sfc_nonods_fine,2
            i_short=(i_long-1)/2 + 1
            ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        end do
        do i_long=2,sfc_nonods_fine,2
            i_short=(i_long-1)/2 + 1
            ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        end do
        
        return 
    end subroutine map_sfc_fine_grid_2_course_grid_vec

! in python:
! a_sfc, b_sfc, ml_sfc, fin_sfc_nonods, nonods_sfc_all_grids, nlevel = best_sfc_mapping_to_sfc_matrix( &
!                                     a, b, ml, &
!                                     fina,cola, ncola, sfc_node_ordering, &
!                                     nonods, max_nonods_sfc_all_grids, max_nlevel) 
    subroutine vector_best_sfc_mapping_to_sfc_matrix_unstructured( &
        vec_a_sfc_all_un,fina_sfc_all_un, &
        cola_sfc_all_un,ncola_sfc_all_un, &
        ndim, & ! ***aded new variables
        vec_b_sfc, ml_sfc, &
        fin_sfc_nonods, nonods_sfc_all_grids, nlevel, &
        vec_a, vec_b, ml,  &
        fina,cola, sfc_node_ordering, ncola, &
        nonods, max_nonods_sfc_all_grids, &
        max_ncola_sfc_all_un, max_nlevel)  
        ! It does this with a kernal size of nfilt_size_sfc. 
        ! this subroutine finds the space filling curve representation of matrix eqns A T=b 
        ! - that is it forms matrix a and vector b and the soln vector is T 
        ! although T is not needed here. 
        ! It does this with a kernal size of nfilt_size_sfc. 
        ! It uses the BEST approach we can to form these tridigonal matrix approximations on different grids. 
        ! It also puts the vector b in space filling curve ordering. 
        ! it forms a series of matricies and vectors on a number of increasing coarse 1d grids 
        ! from nonods in length to 1 in length and stores this matrix in a_sfc. Similarly for the vectors b,ml 
        ! which are stored in b_sfc, ml_sfc. 
        use, intrinsic :: iso_fortran_env
        implicit none
        ! fin_sfc_nonods(ilevel)=the start of course level ilevel and there 
        ! are nlevel grids from course to fine. 
        ! nonods_sfc_all_grids=total number of nodes all in all the grid levels. 
        ! ml is a vector possibly contsining the mass assocated with each cell/node of the original finite mesh. 
        ! sfc_node_ordering(fem node number)=i_sfc_order. Here i_sfc_order is the number of the node meansured along 
        ! the space filling curve trajectory. 
        ! nonods=number of finite element nodes in the mesh.
        ! max_nonods_sfc_all_grids = max number of nodes e.g. use 4*nonods.
        ! max_nlevel= max number of grid levels(e.g.=100). It can also be calculated from the subroutine  
        ! call in python: nlevel = calculate_nlevel_sfc(nonods). 
        !        relax_keep_off=0.7 ! works -how much of the not found value to add into the diagonal of the sfc matrix a_sfc
        ! relax_keep_off=0.0 (dont add any - more stable); relax_keep_off=1.0 (more accurate). =0.5 compromise. 
        !        relax_keep_off=0.5 ! works for hard problem =0.9 not work, =0.7 works, =0.0 works
        !        relax_keep_off=0.75 ! works for hard problem =0.9 not work, =0.8 not work, =0.7 works, =0.0 works
        ! 
        ! fina,cola, ncola are used to define the sparcity pattern of the matrix. 
        ! ncola=number of potentially none-zeros in the nonods*nonods matrix a.  
        ! cola(count)=coln of the matrix a associated with entry count of matrix a - that is a(count).  
        ! fina(inod) start of the inod row of a matrix.
        ! SUPPOSE THE MESH COMPRISSES OF 2 RECTANGULAR ELEMENTS AS BELOW...
        !      1-----2-----3
        !      !     !     !
        !      !     !     !        ndglno(1:4)=4,5,1,2                ndglno(5:8)=5,6,2,3
        !      4-----5-----6
        ! THEN THE MATRIX HAS THE FORM (X is a non-zero entry in the matrix):
        !           1  2  3  4  5  6 - column
        ! row 1    (X  X  0  X  X  0)
        ! row 2    (X  X  X  X  X  X)
        ! row 3    (0  X  X  0  X  X)
        ! row 4    (X  X  0  X  X  0)
        ! row 5    (X  X  X  X  X  X)
        ! row 6    (0  X  X  0  X  X)
        ! The comparact row storage only stores the non-zeros. 
        ! cola(1 )=1, cola(2 )=2, cola(3 )=4, cola(4 )=5,                                *****row 1   fina(1)=1
        ! cola(5 )=1, cola(6 )=2, cola(7 )=3, cola(8 )=4, cola(9 )=5, cola(10)=6,        *****row 2   fina(2)=5
        ! cola(11)=2, cola(12)=3, cola(13)=5, cola(14)=6,                                *****row 3   fina(3)=11
        ! cola(15)=1, cola(16)=2, cola(17)=4, cola(18)=5,                                *****row 4   fina(4)=15
        ! cola(19)=1, cola(20)=2, cola(21)=3, cola(22)=4, cola(23)=5, cola(24)=6,        *****row 5   fina(5)=19
        ! cola(25)=2, cola(26)=3, cola(27)=5, cola(28)=6                                 *****row 6   fina(6)=25
        !                                                                                             fina(7)=29
        ! 
        ! 
        integer, intent( in ) :: ncola, nonods, max_nonods_sfc_all_grids
        integer, intent( in ) :: max_ncola_sfc_all_un, max_nlevel
        integer, intent( in ) :: ndim
        real(real64), intent( out ) :: vec_a_sfc_all_un(ndim,ndim,max_ncola_sfc_all_un), &
        vec_b_sfc(ndim,max_nonods_sfc_all_grids), &
        ml_sfc(max_nonods_sfc_all_grids) 
        real(real64), intent( in ) :: vec_a(ndim,ndim,ncola), vec_b(ndim,nonods), ml(nonods)
        integer, intent( out ) :: nonods_sfc_all_grids, fin_sfc_nonods(max_nlevel+1), nlevel
        integer, intent( out ) :: fina_sfc_all_un(max_nonods_sfc_all_grids+1), &
        cola_sfc_all_un(max_ncola_sfc_all_un),ncola_sfc_all_un
        integer, intent( in ) :: fina(nonods+1), cola(ncola)
        integer, intent( in ) :: sfc_node_ordering(nonods)
        ! local variables...
        integer, allocatable :: sfc_node_ordering_inverse(:), in_row_count(:)
        integer, allocatable :: fina_sfc_all_un2(:)
        integer i, count, count2, nodj, nodi_sfc, ilevel, ilevel2
        integer ifinest_nod_sfc, jfinest_nod_sfc, icourse_nod_sfc, jcourse_nod_sfc
        integer icourse_nod_sfc_all
        integer ifinest_nod, jfinest_nod, ipt
        integer sfc_nonods_fine, sfc_nonods_course, sfc_nonods_accum 
        integer nrow, count_col, idisplace, jcourse_nod_sfc_all2
        integer jcourse_nod_sfc_all, ifinest_nod_sfc_all, count_all
        logical found
        ! 
        ! print *,'2-just inside vector_best_sfc_mapping_to_sfc_matrix_unstructured'
        ! calculate nlevel from nonods
        call calculate_nlevel_sfc(nlevel,nonods)
        ! 
        ! form SFC matrix...
        !        a_sfc(:,:)=0.0
        vec_b_sfc(:,:)=0.0
        ml_sfc(:)=0.0
        do ifinest_nod=1,nonods
            ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
            vec_b_sfc(:,ifinest_nod_sfc)=vec_b(:,ifinest_nod)
            ml_sfc(ifinest_nod_sfc)=ml(ifinest_nod)
        end do 
        !       print *,'here 1 nlevel:',nlevel
        ! 
        ! coarsen...
        sfc_nonods_accum=1
        fin_sfc_nonods(1)=sfc_nonods_accum
        sfc_nonods_accum=sfc_nonods_accum + nonods
        fin_sfc_nonods(2)=sfc_nonods_accum 
        do ilevel=2,nlevel
            !           print *,'ilevel=',ilevel
            sfc_nonods_fine=fin_sfc_nonods(ilevel)-fin_sfc_nonods(ilevel-1)
            if(sfc_nonods_fine.le.1) stop 13331 ! something went wrong. 
            sfc_nonods_course = (sfc_nonods_fine-1)/2 + 1
            !           call map_sfc_course_grid( a_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
            !                                     a_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
            !           print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1):', &
            !                    fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel-1)
            !           print *,'max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine:', &
            !                    max_nonods_sfc_all_grids,sfc_nonods_course,sfc_nonods_fine
            
            call map_sfc_fine_grid_2_course_grid_vec( &
                ml_sfc(fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                ml_sfc(fin_sfc_nonods(ilevel-1)),sfc_nonods_fine)
            !           ipt=fin_sfc_nonods(ilevel)
            !           print *,'ml_sfc(ipt:ipt+sfc_nonods_course-1):',ml_sfc(ipt:ipt+sfc_nonods_course-1)
            !           ipt=fin_sfc_nonods(ilevel-1)
            !           print *,'ml_sfc(ipt:ipt+sfc_nonods_fine-1):',ml_sfc(ipt:ipt+sfc_nonods_fine-1)
            !       print *,'here 1.1' 
            call vector_map_sfc_fine_grid_2_course_grid_vec( &
                vec_b_sfc(:,fin_sfc_nonods(ilevel)),sfc_nonods_course, &
                vec_b_sfc(:,fin_sfc_nonods(ilevel-1)),sfc_nonods_fine, &
                ndim)
            !       print *,'here 1.2' 
            sfc_nonods_accum = sfc_nonods_accum + sfc_nonods_course
            fin_sfc_nonods(ilevel+1)=sfc_nonods_accum
        end do
        nonods_sfc_all_grids=sfc_nonods_accum-1
        if(max_nonods_sfc_all_grids<nonods_sfc_all_grids) then
            print *,'run out of memory here stopping'
            stop 2822
        endif
        ! print *,'here 2'
        !         stop 25
        ! sfc_node_ordering(nod) = new node numbering from current node number nod.
        allocate(sfc_node_ordering_inverse(nonods))
        do ifinest_nod=1,nonods
            ifinest_nod_sfc = sfc_node_ordering(ifinest_nod) 
            sfc_node_ordering_inverse(ifinest_nod_sfc) = ifinest_nod
        end do 
        !     
        ! print *,'---max_ncola_sfc_all_un,ncola,nlevel:',max_ncola_sfc_all_un,ncola,nlevel
        ! print *,'---nonods,nonods_sfc_all_grids:',nonods,nonods_sfc_all_grids
        ! print *,'fin_sfc_nonods(1:,nlevel+1):',fin_sfc_nonods(1:nlevel+1)
        !        a_sfc_all_un=0.0
        count_all=0
        do ilevel=1,nlevel
            ilevel2=2**(ilevel-1)
            ! print *,'--- ilevel=',ilevel
            idisplace = fin_sfc_nonods(ilevel) 
            fina_sfc_all_un(idisplace) = count_all+1
            do ifinest_nod_sfc=1,nonods
                !              print *,'ifinest_nod_sfc=',ifinest_nod_sfc
                ifinest_nod = sfc_node_ordering_inverse(ifinest_nod_sfc)
                icourse_nod_sfc_all = idisplace + (ifinest_nod_sfc-1)/ilevel2
                do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                    jfinest_nod = cola(count)
                    jfinest_nod_sfc = sfc_node_ordering(jfinest_nod) 
                    jcourse_nod_sfc_all = idisplace + (jfinest_nod_sfc-1)/ilevel2
                    ! look to see if we have included jcourse_nod_sfc_all yet
                    found=.false.
                    do count2=fina_sfc_all_un(icourse_nod_sfc_all),count_all
                        jcourse_nod_sfc_all2=cola_sfc_all_un(count2)
                        if(jcourse_nod_sfc_all==jcourse_nod_sfc_all2) then
                            found=.true.
                            vec_a_sfc_all_un(:,:,count2)=vec_a_sfc_all_un(:,:,count2)+vec_a(:,:,count) ! map from original matrix
                        endif
                    end do
                    if(.not.found) then
                        count_all=count_all+1
                        cola_sfc_all_un(count_all) = jcourse_nod_sfc_all
                        !                    a_sfc_all_un(count_all)=a_sfc_all_un(count_all)+a(count) ! map from original matrix
                        vec_a_sfc_all_un(:,:,count_all)=vec_a(:,:,count) ! map from original matrix
                    endif
                end do ! do count=fina(ifinest_nod),fina(ifinest_nod+1)-1
                fina_sfc_all_un(icourse_nod_sfc_all+1) = count_all+1
            end do ! do ifinest_nod_sfc=1,nonods
            ! print *,'here2'
            ! print *,'fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1):', &
            ! fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)
            ! print *,'diff=',fin_sfc_nonods(ilevel+1)-fin_sfc_nonods(ilevel)
            ! print *,'fina_sfc_all_un(fin_sfc_nonods(ilevel)):',fina_sfc_all_un(fin_sfc_nonods(ilevel))
            ! print *,'fina_sfc_all_un(fin_sfc_nonods(ilevel+1))-1:',fina_sfc_all_un(fin_sfc_nonods(ilevel+1))-1
            ! print *,'difference:', &
            ! fina_sfc_all_un(fin_sfc_nonods(ilevel+1)) - fina_sfc_all_un(fin_sfc_nonods(ilevel))
            if(.false.) then
                do icourse_nod_sfc_all=fin_sfc_nonods(ilevel),fin_sfc_nonods(ilevel+1)-1
                    print *,'icourse_nod_sfc_all=',icourse_nod_sfc_all
                    print *,'cola_sfc_all_un(count2):', &
                    (cola_sfc_all_un(count2),count2=fina_sfc_all_un(icourse_nod_sfc_all), &
                    fina_sfc_all_un(icourse_nod_sfc_all+1)-1) 
                    print *,'vec_a_sfc_all_un(count2):', &
                    (vec_a_sfc_all_un(:,:,count2),count2=fina_sfc_all_un(icourse_nod_sfc_all), &
                    fina_sfc_all_un(icourse_nod_sfc_all+1)-1) 
                end do
            endif
        end do ! do ilevel=1,nlevel
        
        ncola_sfc_all_un = fina_sfc_all_un(nonods_sfc_all_grids+1)-1
        if( max_ncola_sfc_all_un < ncola_sfc_all_un ) then
            print *,'run out of memory here stopping'
            stop 2825
        endif
        ! 
        ! print *,'ncola, ncola_sfc_all_un:',ncola, ncola_sfc_all_un
        ! print *,'nonods_sfc_all_grids:',nonods_sfc_all_grids
        ! print *,'fina_sfc_all_un(nonods_sfc_all_grids+1)-fina_sfc_all_un(nonods_sfc_all_grids):', &
        ! fina_sfc_all_un(nonods_sfc_all_grids+1)-fina_sfc_all_un(nonods_sfc_all_grids)
        
        ! !        stop 282
        ! print *,'just leaving vector_best_sfc_mapping_to_sfc_matrix_unstructured'
        return 
    end subroutine vector_best_sfc_mapping_to_sfc_matrix_unstructured

    subroutine vector_map_sfc_fine_grid_2_course_grid_vec( vec_ml_sfc_course,sfc_nonods_course, &
        vec_ml_sfc_fine,sfc_nonods_fine, &
        ndim)
        use, intrinsic :: iso_fortran_env
        implicit none 
        integer, intent( in ) :: sfc_nonods_course, sfc_nonods_fine
        integer, intent( in ) :: ndim
        real(real64), intent( out ) :: vec_ml_sfc_course(ndim,sfc_nonods_course)
        real(real64), intent( in ) :: vec_ml_sfc_fine(ndim,sfc_nonods_fine)
        ! local variables...
        integer i_short, i_long
        ! 
        vec_ml_sfc_course(:,:)=0.0
        !        do i_short=1,sfc_nonods_course
        !           i_long=(i_short-1)*2 + 1
        !           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        
        !           i_long=min( (i_short-1)*2 + 2, sfc_nonods_fine)----miss this out---
        !           ml_sfc_course(i_short)=ml_sfc_course(i_short)+ml_sfc_fine(i_long)
        !        end do
        ! 
        do i_long=1,sfc_nonods_fine,2
            i_short=(i_long-1)/2 + 1
            vec_ml_sfc_course(:,i_short)=vec_ml_sfc_course(:,i_short)+vec_ml_sfc_fine(:,i_long)
        end do
        do i_long=2,sfc_nonods_fine,2
            i_short=(i_long-1)/2 + 1
            vec_ml_sfc_course(:,i_short)=vec_ml_sfc_course(:,i_short)+vec_ml_sfc_fine(:,i_long)
        end do
        
        return 
    end subroutine vector_map_sfc_fine_grid_2_course_grid_vec