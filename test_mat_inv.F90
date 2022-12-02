program main 
implicit none 
real, dimension(8,8) :: a ,ainv
integer:: n=8, nmax = 8
real, dimension(8,8) :: mat, mat2 
real, dimension(8) :: x, b 

a = reshape( (/  &
0.814723686393179, 0.905791937075619, 0.126986816293506, 0.913375856139019, 0.632359246225410, &
0.097540404999410, 0.278498218867048, 0.546881519204984, 0.957506835434298, 0.964888535199277, &
0.157613081677548, 0.970592781760616, 0.957166948242946, 0.485375648722841, 0.800280468888800, &
0.141886338627215, 0.421761282626275, 0.915735525189067, 0.792207329559554, 0.959492426392903, &
0.655740699156587, 0.035711678574190, 0.849129305868777, 0.933993247757551, 0.678735154857773, &
0.757740130578333, 0.743132468124916, 0.392227019534168, 0.655477890177557, 0.171186687811562, &
0.706046088019609, 0.031832846377421, 0.276922984960890, 0.046171390631154, 0.097131781235848, &
0.823457828327293, 0.694828622975817, 0.317099480060861, 0.950222048838355, 0.034446080502909, &
0.438744359656398, 0.381558457093008, 0.765516788149002, 0.795199901137063, 0.186872604554379, &
0.489764395788231, 0.445586200710899, 0.646313010111265, 0.709364830858073, 0.754686681982361 , &
0.276025076998578, 0.679702676853675, 0.655098003973841, 0.162611735194631, 0.118997681558377, &
0.498364051982143, 0.959743958516081, 0.340385726666133, 0.585267750979777, 0.223811939491137, &
0.751267059305653, 0.255095115459269, 0.505957051665142, 0.699076722656686/), &
(/8,8/))

print *, a
ainv=a
call matinv(ainv,n,nmax,mat,mat2,x,b)
print *, ainv 
print *, matmul(ainv,a)


    end program


    SUBROUTINE MATINV(A,N,NMAX,MAT,MAT2,X,B)
        ! This sub finds the inverse of the matrix A and puts it back in A. 
        ! MAT, MAT2 & X,B are working vectors. 
               IMPLICIT NONE
               INTEGER N,NMAX
               REAL A(NMAX,NMAX),MAT(N,N),MAT2(N,N),X(N),B(N)
        ! Local variables
               INTEGER ICOL,IM,JM
        
        
                 DO IM=1,N
                   DO JM=1,N
                     MAT(IM,JM)=A(IM,JM)
                   END DO
                 END DO
        ! Solve MAT X=B (NB MAT is overwritten).  
               CALL SMLINN_FACTORIZE(MAT,X,B,N,N)
        !
               DO ICOL=1,N
        !
        ! Form column ICOL of the inverse. 
                 DO IM=1,N
                   B(IM)=0.
                 END DO
                 B(ICOL)=1.0
        ! Solve MAT X=B (NB MAT is overwritten).  
               CALL SMLINN_SOLVE_LU(MAT,X,B,N,N)
        ! X contains the column ICOL of inverse
                 DO IM=1,N
                   MAT2(IM,ICOL)=X(IM)
                 END DO 
        !
              END DO
        !
        ! Set A to MAT2
                 DO IM=1,N
                   DO JM=1,N
                     A(IM,JM)=MAT2(IM,JM)
                   END DO
                 END DO
               RETURN
               END SUBROUTINE MATINV
        !
        !
        !     
              
                SUBROUTINE SMLINN_FACTORIZE(A,X,B,NMX,N)
                IMPLICIT NONE
                INTEGER NMX,N
                REAL A(NMX,NMX),X(NMX),B(NMX)
                REAL R
                INTEGER K,I,J
        !     Form X = A^{-1} B
        !     Useful subroutine for inverse
        !     This sub overwrites the matrix A. 
                DO K=1,N-1
                   DO I=K+1,N
                      A(I,K)=A(I,K)/A(K,K)
                   END DO
                   DO J=K+1,N
                      DO I=K+1,N
                         A(I,J)=A(I,J) - A(I,K)*A(K,J)
                      END DO
                   END DO
                END DO
        !     
              if(.false.) then
        !     Solve L_1 x=b
                DO I=1,N
                   R=0.
                   DO J=1,I-1
                      R=R+A(I,J)*X(J)
                   END DO
                   X(I)=B(I)-R
                END DO
        !     
        !     Solve U x=y
                DO I=N,1,-1
                   R=0.
                   DO J=I+1,N
                      R=R+A(I,J)*X(J)
                   END DO
                   X(I)=(X(I)-R)/A(I,I)
                END DO
              endif
                RETURN
                END SUBROUTINE SMLINN_FACTORIZE
        !     
        !     
              
                SUBROUTINE SMLINN_SOLVE_LU(A,X,B,NMX,N)
                IMPLICIT NONE
                INTEGER NMX,N
                REAL A(NMX,NMX),X(NMX),B(NMX)
                REAL R
                INTEGER K,I,J
        !     Form X = A^{-1} B
        !     Useful subroutine for inverse
        !     This sub overwrites the matrix A. 
               if(.false.) then
                DO K=1,N-1
                   DO I=K+1,N
                      A(I,K)=A(I,K)/A(K,K)
                   END DO
                   DO J=K+1,N
                      DO I=K+1,N
                         A(I,J)=A(I,J) - A(I,K)*A(K,J)
                      END DO
                   END DO
                END DO
               endif
        !     
        !     Solve L_1 x=b
                DO I=1,N
                   R=0.
                   DO J=1,I-1
                      R=R+A(I,J)*X(J)
                   END DO
                   X(I)=B(I)-R
                END DO
        !     
        !     Solve U x=y
                DO I=N,1,-1
                   R=0.
                   DO J=I+1,N
                      R=R+A(I,J)*X(J)
                   END DO
                   X(I)=(X(I)-R)/A(I,I)
                END DO
                RETURN
                END SUBROUTINE SMLINN_SOLVE_LU