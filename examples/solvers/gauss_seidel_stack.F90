module gauss_seidel_stack_mod
	implicit none
	
contains

  subroutine main()
      real :: data(256, 256)
      integer :: i, j, k

      do k = 1, 1000, 1
        do i = 2, 255, 1
          do j = 2, 255, 1
            data(j,i) = (data(j,i - 1) + data(j,i + 1) + data(j - 1,i) + data(j + 1,i)) * 0.25
          enddo
        enddo
      enddo

  end subroutine main
end module gauss_seidel_stack_mod

program me
  use gauss_seidel_stack_mod
	
implicit none

  call main()
end program me
