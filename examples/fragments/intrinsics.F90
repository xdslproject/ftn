module intrinsics_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

contains

  subroutine calc()
    call test_transpose()
    call test_matmul()
  end subroutine calc

  subroutine test_transpose()
    integer :: a(10,10), b(10,10)
    integer, dimension(:,:), allocatable :: c, d

    integer :: i, j

    allocate(c(10, 10), d(10, 10))

    do i=1, 10
      do j=1, 10
        a(j,i)=j
        c(j,i)=j
      end do
    end do

    b=transpose(a)
    d=transpose(c)

    do i=1, 10
      do j=1, 10
        call assert(b(j,i)==i, __FILE__, __LINE__)
        call assert(d(j,i)==i, __FILE__, __LINE__)
      end do
    end do
  end subroutine test_transpose

  subroutine test_matmul()
    real :: a(10,10), b(10,10), c(10,10)
    real, dimension(:,:), allocatable :: d, e, f

    integer :: i, j

    do i=1, 10
      do j=1, 10
        a(j,i)=real(j)
        b(i,j)=real(i)
      end do
    end do

    c=matmul(a, b)

    call compare_matmul(a, b, c, 10, 10, 10)

    allocate(d(10,10), e(10,10), f(10,10))

    do i=1, 10
      do j=1, 10
        d(j,i)=real(j)
        e(i,j)=real(i)
      end do
    end do

    f=matmul(a, b)

    call compare_matmul(d, e, f, 10, 10, 10)
    deallocate(d, e, f)
  end subroutine test_matmul

  subroutine compare_matmul(a, b, c, size_a_one, size_a_two, size_b_two)
    real, dimension(:,:), intent(in) :: a, b, c
    integer, intent(in) :: size_a_one, size_a_two, size_b_two
    integer :: i, j, k
    real :: comparison

    do i=1, size_a_one
      do j=1, size_b_two
        comparison=0.0
        do k=1, size_a_two
          comparison = comparison + (a(k,i) * b(j,k))
        end do
        call assert(c(j,i)==comparison, __FILE__, __LINE__)
      end do
    end do
  end subroutine compare_matmul

end module intrinsics_test

#ifndef FRAGMENT_ONLY
program driver
  use intrinsics_test
  implicit none

  call assert_init(.false.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
