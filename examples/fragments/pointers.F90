module pointers_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

  real, dimension(:), pointer :: ptr3=>NULL()

contains

  subroutine calc()
      real, dimension(:), allocatable, target :: a, b, z, x
      integer, dimension(:,:,:), allocatable, target :: c
      real, dimension(:), pointer :: ptr1=>NULL(), ptr2=>NULL()
      integer, dimension(:,:,:), pointer :: ptr_md=>NULL()
      real :: t

      integer :: i, j, k

      allocate(a(100), b(100), c(10,5,15), z(10), x(10))

      ptr1 => z
      ptr2 => x

      ptr1=(/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 /)
      ptr2=(/ (real(i), i = 11, 20) /)

      do i=1, 10
        call assert(ptr1(i)==real(i), __FILE__, __LINE__)
        call assert(z(i)==real(i), __FILE__, __LINE__)
        call assert(ptr2(i)==real(i+10), __FILE__, __LINE__)
        call assert(x(i)==real(i+10), __FILE__, __LINE__)
      end do

      ! Test assignment of underlying allocatable pointed to, as equals
      ! and not =>, therefore the array x should now hold values of z
      ptr2=ptr1
      do i=1, 10
        call assert(ptr2(i)==real(i), __FILE__, __LINE__)
        call assert(x(i)==real(i), __FILE__, __LINE__)
      end do


      do i=1, 100
        a(i)=i
        b(i)=100-i
      end do

      ! Test pointing to a target
      ptr1 => a
      call assert(rank(ptr1)==1, __FILE__, __LINE__)
      call assert(size(ptr1, 1)==100, __FILE__, __LINE__)
      call assert(size(ptr1)==100, __FILE__, __LINE__)
      do i=1, 100
        call assert(ptr1(i)==real(i), __FILE__, __LINE__)
      end do

      ! Test another pointer pointing to a target
      ptr2 => a
      call assert(rank(ptr2)==1, __FILE__, __LINE__)
      call assert(size(ptr2, 1)==100, __FILE__, __LINE__)
      call assert(size(ptr2)==100, __FILE__, __LINE__)
      do i=1, 100
        call assert(ptr2(i)==real(i), __FILE__, __LINE__)
      end do

      ! Test assigning pointer to a pointer
      ptr3 => ptr1
      call assert(rank(ptr3)==1, __FILE__, __LINE__)
      call assert(size(ptr3, 1)==100, __FILE__, __LINE__)
      call assert(size(ptr3)==100, __FILE__, __LINE__)
      do i=1, 100
        call assert(ptr3(i)==real(i), __FILE__, __LINE__)
      end do

      ! Assign to ptr1, but as point to a ensure all have this value
      ptr1(20)=34
      call assert(a(20)==real(34), __FILE__, __LINE__)
      call assert(ptr1(20)==real(34), __FILE__, __LINE__)
      call assert(ptr2(20)==real(34), __FILE__, __LINE__)
      call assert(ptr3(20)==real(34), __FILE__, __LINE__)

      ! Test assigning ptr2 to another target, pointers must have different values
      ptr2 => b
      do i=1, 100
        if (i == 20) then
          call assert(ptr1(i)==real(34), __FILE__, __LINE__)
        else
          call assert(ptr1(i)==real(i), __FILE__, __LINE__)
        end if
        call assert(ptr2(i)==real(100-i), __FILE__, __LINE__)
      end do

      ! Tests calling a procedure with pointers (which will swap them) and then
      ! ensure that the pointers are swapped around. Main test here is to ensure
      ! a procedure can be called with pointers
      call swap(ptr1, ptr2)
      do i=1, 100
        if (i == 20) then
          call assert(ptr2(i)==real(34), __FILE__, __LINE__)
        else
          call assert(ptr2(i)==real(i), __FILE__, __LINE__)
        end if
        call assert(ptr1(i)==real(100-i), __FILE__, __LINE__)
      end do

      ! Grab a scalar from a pointer and store it
      t=ptr1(3)
      call assert(t==real(100-3), __FILE__, __LINE__)

      call modify_array_ptr_one(ptr1, 20, 3.1)
      call assert(ptr1(20)==3.1, __FILE__, __LINE__)

      call modify_array_one(ptr1, 19, 87.64)
      call assert(ptr1(19)==87.64, __FILE__, __LINE__)

      call modify_array_two(ptr1, 76, 992.32)
      call assert(ptr1(76)==992.32, __FILE__, __LINE__)

      ! Test pointer to multi-dimensional array
      do i=1, 15
        do j=1, 5
          do k=1, 10
            c(k, j, i) = k+(j*10)+(i*100)
          end do
        end do
      end do
      ptr_md=>c
      call assert(rank(ptr_md)==3, __FILE__, __LINE__)
      call assert(size(ptr_md)==750, __FILE__, __LINE__)
      call assert(size(ptr_md, 1)==10, __FILE__, __LINE__)
      call assert(size(ptr_md, 2)==5, __FILE__, __LINE__)
      call assert(size(ptr_md, 3)==15, __FILE__, __LINE__)

      call assert(ptr_md(3,4,5)==543, __FILE__, __LINE__)
      call assert(ptr_md(8,5,15)==1558, __FILE__, __LINE__)

      call modify_3darray_ptr_one(ptr_md, 2, 3, 4, 100)
      call assert(ptr_md(2,3,4)==100, __FILE__, __LINE__)

      call modify_3darray_one(ptr_md, 4, 4, 7, 87)
      call assert(ptr_md(4,4,7)==87, __FILE__, __LINE__)

      call modify_3darray_two(ptr_md, 7, 1, 3, 13)
      call assert(ptr_md(7,1,3)==13, __FILE__, __LINE__)

      deallocate(a, b, c, z, x)
  end subroutine calc

  subroutine swap(swp1, swp2)
    real, dimension(:), pointer :: swp1, swp2

    real, dimension(:), pointer :: swp_tmp

    swp_tmp => swp1
    swp1 => swp2
    swp2 => swp_tmp
  end subroutine swap

  subroutine modify_array_ptr_one(a, idx, value)
    real, dimension(:), pointer, intent(inout) :: a
    integer, intent(in) :: idx
    real, intent(in) :: value

    a(idx)=value
  end subroutine modify_array_ptr_one

  subroutine modify_3darray_ptr_one(array, k, j, i, value)
    integer, dimension(:,:,:), pointer, intent(inout) :: array
    integer, intent(in) :: i, j, k, value

    array(k, j, i)=value
  end subroutine modify_3darray_ptr_one

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
end module pointers_test

#ifndef FRAGMENT_ONLY
program driver
  use pointers_test
  implicit none

  call assert_init(.false.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
