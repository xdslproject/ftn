! Tests OpenMP target directive with simd

module ex5_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a, b, c

    integer :: i

    allocate(a(100), b(100), c(100))

    !$omp target simd simdlen(16)
    do i=1, 100
      c(i)=a(i)+b(i)
    end do
    !$omp end target simd
  end subroutine calc

end module ex5_test

program main
  use ex5_test

implicit none

  call calc()
end program main

