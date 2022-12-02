program matmultest
    implicit none
    real, dimension(5,5) :: a,b,c 
    real, dimension(5) :: d,e

    a=reshape( &
        (/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25/), &
        (/5,5/) ) 
    b=reshape( &
        (/26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50/), &
        (/5,5/) ) 
    c = matmul(a,b)
    d=(/101,102,103,104,105/)
    e = matmul(a,d)
    print *, 'a', a 
    print *, 'b', b 
    print *, 'c', c 
    print *, 'd', d 
    print *, 'e', e
    print *, 'a*b(:,1)',matmul(a,b(:,1))
    
end
