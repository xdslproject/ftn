! Tests OpenMP target data directive with target update

module ex7_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a, b, c

    integer :: i

    allocate(a(100), b(100), c(100))

    !$omp target data map(from:c)
    !$omp target map(to:a,b)
    do i=1, 100
      c(i)=a(i)+b(i)
    end do
    !$omp end target

    !$omp target update from(c)

    !$omp end target data
  end subroutine calc

end module ex7_test

program main
  use ex7_test

implicit none

  call calc()
end program main
