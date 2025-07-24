module allocatables_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

      real, dimension(:), allocatable :: global_array

contains

  subroutine calc()
      real, dimension(:), allocatable :: a, b, tmp
      integer, dimension(:,:,:), allocatable :: c

      integer :: i, j, k

      allocate(a(100), b(100), global_array(100), c(10,10,10))

      ! Test that the rank is correctly returned (number of dimensions)
      call assert(rank(a)==1, __FILE__, __LINE__)
      call assert(rank(b)==1, __FILE__, __LINE__)
      call assert(rank(global_array)==1, __FILE__, __LINE__)
      call assert(rank(c)==3, __FILE__, __LINE__)

      ! Test that size works, both in total and per dimension
      call assert(size(a, 1)==100, __FILE__, __LINE__)
      call assert(size(a)==100, __FILE__, __LINE__)
      call assert(size(global_array)==100, __FILE__, __LINE__)
      call assert(size(c)==1000, __FILE__, __LINE__)
      call assert(size(c, 2)==10, __FILE__, __LINE__)

      do i=1, 100
        a(i)=real(i)
        b(i)=real(100-i)
        global_array(i)=real(i*10)
      end do

      ! Test values
      do i=1, 100
        call assert(a(i)==real(i), __FILE__, __LINE__)
        call assert(b(i)==real(100-i), __FILE__, __LINE__)
        call assert(global_array(i)==real(i*10), __FILE__, __LINE__)
      end do

      ! Assign to allocatables and ensure value is set
      a(20)=34.5
      b(50)=165.2
      global_array(70)=23.1
      call assert(a(20)==34.5, __FILE__, __LINE__)
      call assert(b(50)==165.2, __FILE__, __LINE__)
      call assert(global_array(70)==23.1, __FILE__, __LINE__)

      ! Call a procedure to modify the changed values,
      ! tests passing allocatables to procedures
      ! First procedure is unknown array and second
      ! sized array, tests we can pass to both of these
      call modify_array_one(a, 20, 20.0)
      call modify_array_two(b, 50, 50.0)
      call modify_array_two(global_array, 70, 700.0)
      do i=1, 100
        call assert(a(i)==real(i), __FILE__, __LINE__)
        call assert(b(i)==real(100-i), __FILE__, __LINE__)
        call assert(global_array(i)==real(i*10), __FILE__, __LINE__)
      end do

      call modify_array_one(global_array, 60, 123.4)
      call assert(global_array(60)==123.4, __FILE__, __LINE__)

      ! Test move alloc, swapping the memory associated with
      ! these allocatables to swap them around
      call move_alloc(a, tmp)
      call move_alloc(b, a)
      call move_alloc(tmp, b)
      do i=1, 100
        call assert(b(i)==real(i), __FILE__, __LINE__)
        call assert(a(i)==real(100-i), __FILE__, __LINE__)
      end do

      call modify_array_three(a, 80, 13.4)
      call assert(a(80) == 13.4, __FILE__, __LINE__)

      ! Test multi-dimensional arrays
      do i=1, 10
        do j=1, 10
          do k=1, 10
            c(k, j, i) = k+(j*10)+(i*100)
          end do
        end do
      end do
      call assert(c(3,4,5)==543, __FILE__, __LINE__)
      call assert(c(8,9,1)==198, __FILE__, __LINE__)

      call modify_3darray_one(c, 2, 3, 4, 100)
      call assert(c(2,3,4)==100, __FILE__, __LINE__)
      call modify_3darray_two(c, 6, 7, 8, 200)
      call assert(c(6,7,8)==200, __FILE__, __LINE__)
      call modify_3darray_three(c, 4, 5, 6, 300)
      call assert(c(4,5,6)==300, __FILE__, __LINE__)

      deallocate(a,b,c)
  end subroutine calc

  subroutine modify_array_one(a, idx, value)
    real, dimension(:), intent(inout) :: a
    integer, intent(in) :: idx
    real, intent(in) :: value

    a(idx)=value
  end subroutine modify_array_one

  subroutine modify_array_two(a, idx, value)
    real, dimension(100), intent(inout) :: a
    integer, intent(in) :: idx
    real, intent(in) :: value

    a(idx)=value
  end subroutine modify_array_two

  subroutine modify_array_three(a, idx, value)
    real, dimension(:), allocatable, intent(inout) :: a
    integer, intent(in) :: idx
    real, intent(in) :: value

    a(idx)=value
  end subroutine modify_array_three

  subroutine modify_3darray_one(array, k, j, i, value)
    integer, dimension(:,:,:), intent(inout) :: array
    integer, intent(in) :: i, j, k, value

    array(k, j, i)=value
  end subroutine modify_3darray_one

  subroutine modify_3darray_two(array, k, j, i, value)
    integer, dimension(10,10,10), intent(inout) :: array
    integer, intent(in) :: i, j, k, value

    array(k, j, i)=value
  end subroutine modify_3darray_two

  subroutine modify_3darray_three(array, k, j, i, value)
    integer, dimension(:,:,:), allocatable, intent(inout) :: array
    integer, intent(in) :: i, j, k, value

    array(k, j, i)=value
  end subroutine modify_3darray_three
end module allocatables_test

#ifndef FRAGMENT_ONLY
program driver
  use allocatables_test
  implicit none

  call assert_init(.false.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
