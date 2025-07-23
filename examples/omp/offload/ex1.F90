! Tests simple OpenMP target directive

module ex1_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a, b, c

    integer :: i

    allocate(a(100), b(100), c(100))

    !$omp target
    do i=1, 100
      c(i)=a(i)+b(i)
    end do
    !$omp end target
  end subroutine calc

end module ex1_test

program main
  use ex1_test

implicit none

  call calc()
end program main
