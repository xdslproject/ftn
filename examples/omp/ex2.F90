! Tests OpenMP parallel do and reduction clause

module ex2_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a

    integer :: i

    real :: result

    allocate(a(100))

    !$omp parallel do
    do i=1, 100
      a(i)=i
    end do
    !$omp end parallel do

    !$omp parallel do reduction(+:result)
    do i=1, 100
      result=result+a(i)
    end do
    !$omp end parallel do
  end subroutine calc

end module ex2_test

program main
  use ex2_test

implicit none

  call calc()
end program main

