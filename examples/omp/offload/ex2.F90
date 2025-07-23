! Tests OpenMP target directive with distribute and teams

module ex2_test
  implicit none

contains

  subroutine calc()
    real, dimension(:), allocatable :: a, b, c

    integer :: i

    allocate(a(100), b(100), c(100))

    !$omp target teams distribute num_teams(3)
    do i=1, 100
      c(i)=a(i)+b(i)
    end do
    !$omp end target teams distribute
  end subroutine calc

end module ex2_test

program main
  use ex2_test

implicit none

  call calc()
end program main

