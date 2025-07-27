module arrays_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

      real :: global_array(100)

contains

  subroutine calc()
      real, dimension(100) :: a
      real :: b(100), t
      real :: z(10), x(10)
      integer :: c(10,5,15)

      integer :: i, j, k

      z=(/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 /)
      x=(/ (real(i), i = 11, 20) /)

      do i=1, 10
        call assert(z(i)==real(i), __FILE__, __LINE__)
        call assert(x(i)==real(i+10), __FILE__, __LINE__)
      end do

      ! Test that the rank is correctly returned (number of dimensions)
      call assert(rank(a)==1, __FILE__, __LINE__)
      call assert(rank(b)==1, __FILE__, __LINE__)
      call assert(rank(global_array)==1, __FILE__, __LINE__)
      call assert(rank(c)==3, __FILE__, __LINE__)

      ! Test that size works, both in total and per dimension
      call assert(size(a, 1)==100, __FILE__, __LINE__)
      call assert(size(a)==100, __FILE__, __LINE__)
      call assert(size(global_array)==100, __FILE__, __LINE__)
      call assert(size(c)==750, __FILE__, __LINE__)
      call assert(size(c, 1)==10, __FILE__, __LINE__)
      call assert(size(c, 2)==5, __FILE__, __LINE__)
      call assert(size(c, 3)==15, __FILE__, __LINE__)

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

      ! Revert global array location 60 back as we are going to use this again
      global_array(60)=600.0

      ! Test entire array copying via an assignment
      a=b
      b=global_array
      do i=1, 100
        call assert(a(i)==real(100-i), __FILE__, __LINE__)
        call assert(b(i)==real(i*10), __FILE__, __LINE__)
      end do

      ! Test multi-dimensional arrays
      do i=1, 15
        do j=1, 5
          do k=1, 10
            c(k, j, i) = k+(j*10)+(i*100)
          end do
        end do
      end do
      call assert(c(3,4,5)==543, __FILE__, __LINE__)
      call assert(c(8,5,15)==1558, __FILE__, __LINE__)

      call modify_3darray_one(c, 2, 3, 4, 100)
      call assert(c(2,3,4)==100, __FILE__, __LINE__)
      call modify_3darray_two(c, 6, 2, 12, 200)
      call assert(c(6,2,12)==200, __FILE__, __LINE__)
  end subroutine calc

  subroutine modify_array_one(a, idx, value)
    real, dimension(100), intent(inout) :: a
    integer, intent(in) :: idx
    real, intent(in) :: value

    a(idx)=value
  end subroutine modify_array_one

  subroutine modify_array_two(a, idx, value)
    real, dimension(:), intent(inout) :: a
    integer, intent(in) :: idx
    real, intent(in) :: value

    a(idx)=value
  end subroutine modify_array_two

  subroutine modify_3darray_one(array, k, j, i, value)
    integer, dimension(:,:,:), intent(inout) :: array
    integer, intent(in) :: i, j, k, value

    array(k, j, i)=value
  end subroutine modify_3darray_one

  subroutine modify_3darray_two(array, k, j, i, value)
    integer, dimension(10,5,15), intent(inout) :: array
    integer, intent(in) :: i, j, k, value

    array(k, j, i)=value
  end subroutine modify_3darray_two
end module arrays_test

#ifndef FRAGMENT_ONLY
program driver
  use arrays_test
  implicit none

  call assert_init(.false.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
