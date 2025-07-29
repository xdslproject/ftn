module intrinsics_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

contains

  subroutine calc()
    call test_transpose()
    call test_matmul()
    call test_sum()
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

    deallocate(c, d)
  end subroutine test_transpose

  subroutine test_matmul()
    real :: a(5,5), b(5,5), c(5,5)
    real, dimension(:,:), allocatable :: d, e, f

    integer :: i, j

    do i=1, 5
      do j=1, 5
        a(j,i)=real(j)
        b(i,j)=real(i)
      end do
    end do

    c=matmul(a, b)

    call compare_matmul(a, b, c, 5, 5, 5)
    allocate(d(10,10), e(10,10), f(10,10))

    do i=1, 10
      do j=1, 10
        d(j,i)=real(j)
        e(i,j)=real(i)
      end do
    end do

    f=matmul(d, e)

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

  subroutine test_sum()
    real :: stack_data(10,5), out_stack_one(10), out_stack_two(5), out_stack_three
    integer, dimension(:,:), allocatable :: heap_data
    integer, dimension(:), allocatable :: out_heap_one, out_heap_two
    integer i, j, out_heap_three

    allocate(heap_data(10, 5), out_heap_one(10), out_heap_two(5))

    do i=1, 5
      do j=1, 10
        stack_data(j,i)=j
        heap_data(j,i)=j
      end do
    end do

    out_stack_one=sum(stack_data, 2)
    out_stack_two=sum(stack_data, 1)
    out_stack_three=sum(stack_data)

    out_heap_one=sum(heap_data, 2)
    out_heap_two=sum(heap_data, 1)
    out_heap_three=sum(heap_data)

    do j=1, 10
      if (j .le. 5) then
        call assert(out_stack_two(j)==55.0, __FILE__, __LINE__)
        call assert(out_heap_two(j)==55.0, __FILE__, __LINE__)
      end if
      call assert(out_stack_one(j)==real(j*5), __FILE__, __LINE__)
      call assert(out_heap_one(j)==real(j*5), __FILE__, __LINE__)
    end do

    do j=1, 5
      call assert(out_stack_two(j)==55.0, __FILE__, __LINE__)
    end do

    call assert(out_stack_three == 275.0, __FILE__, __LINE__)
    call assert(out_heap_three == 275.0, __FILE__, __LINE__)

    deallocate(heap_data, out_heap_one, out_heap_two)
  end subroutine test_sum

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
