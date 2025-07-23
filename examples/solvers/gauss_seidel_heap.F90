module gauss_seidel_heap_mod
	implicit none
	
contains

	subroutine main()
    	real, dimension(:,:,:), allocatable :: data
    	integer :: i, j, k, m
    	
    	allocate(data(512, 512, 1024))
	
    	do k = 1, 1, 1
      	do m = 2, 1023, 1
        	do j = 2, 511, 1
        		do i = 2, 511, 1
          		data(i,j,m) = (data(i,j-1,m) + data(i,j+1,m) + data(i-1,j,m) + data(i+1,j,m) + &
          						data(i,j,m-1) + data(i,j,m+1)) * 0.16666
          	enddo
        	enddo
      	enddo
    	enddo
	end subroutine main
end module gauss_seidel_heap_mod

program me
  use gauss_seidel_heap_mod
  
implicit none  

	call main()
end program me
