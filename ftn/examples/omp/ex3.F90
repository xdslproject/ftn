! Tests OpenMP target directive with distribute, teams and parallel do

module ftn_example
	implicit none
	
contains

	subroutine calc()			
		real, dimension(:), allocatable :: a, b, c						
		
		integer :: i
		
		allocate(a(100), b(100), c(100))
	
		!$omp target teams distribute parallel do num_teams(3)  
		do i=1, 100
			c(i)=a(i)+b(i)			
		end do
		!$omp end target teams distribute parallel do
	end subroutine calc
	
end module ftn_example

program main
	use ftn_example
	
implicit none
	
	call calc()	
end program main

