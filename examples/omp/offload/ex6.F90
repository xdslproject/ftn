! Tests OpenMP target directive with simd and parallel do

module ex6_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a, b, c

    integer :: i

    allocate(a(100), b(100), c(100))

    !$omp target parallel do simd simdlen(16) num_threads(20)
    do i=1, 100
      c(i)=a(i)+b(i)
    end do
    !$omp end target parallel do simd
  end subroutine calc

end module ex6_test

program main
  use ex6_test

implicit none

  call calc()
end program main

