! Tests OpenMP target enter and exit data

module ex8_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a
    real :: b(100), c(100)

    integer :: i

    allocate(a(100))

    !$omp target enter data map(alloc: c)
    !$omp target map(to:a,b)
    do i=1, 100
      c(i)=a(i)+b(i)
    end do
    !$omp end target

    !$omp target exit data map(from: c)
  end subroutine calc

end module ex8_test

program main
  use ex8_test

implicit none

  call calc()
end program main
