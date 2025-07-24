module pointers_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

  real, dimension(:), pointer :: ptr3=>NULL()

contains

  subroutine calc()
      real, dimension(:), allocatable, target :: a, b
      integer, dimension(:,:,:), allocatable, target :: c
      real, dimension(:), pointer :: ptr1=>NULL(), ptr2=>NULL()
      integer, dimension(:,:,:), pointer :: ptr_md=>NULL()
      real :: t

      integer :: i, j, k

      allocate(a(100), b(100), c(10,10,10))

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

      ! Test pointer to multi-dimensional array
      do i=1, 10
        do j=1, 10
          do k=1, 10
            c(k, j, i) = k+(j*10)+(i*100)
          end do
        end do
      end do
      ptr_md=>c
      call assert(rank(ptr_md)==3, __FILE__, __LINE__)
      call assert(size(ptr_md)==1000, __FILE__, __LINE__)
      call assert(size(ptr_md, 2)==10, __FILE__, __LINE__)
      call assert(ptr_md(3,4,5)==543, __FILE__, __LINE__)
      call assert(ptr_md(8,9,1)==198, __FILE__, __LINE__)

      call modify_3darray_ptr_one(ptr_md, 2, 3, 4, 100)
      call assert(ptr_md(2,3,4)==100, __FILE__, __LINE__)
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
