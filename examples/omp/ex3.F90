! Tests OpenMP parallel do with collapse clause

module ex3_test
  implicit none

contains

  subroutine calc()
    real, dimension(:,:), allocatable :: a

    integer :: i, j

    allocate(a(100,100))

    !$omp parallel do collapse(2)
    do i=1, 100
      do j=1, 100
        a(j,i)=i*j
      end do
    end do
    !$omp end parallel do
  end subroutine calc

end module ex3_test

program main
  use ex3_test

implicit none

  call calc()
end program main

