#ifndef FRAGMENT_ONLY
module pointer_test
  use assertion
#endif
  implicit none

  real, dimension(:), pointer :: ptr3=>NULL()

contains

  subroutine calc()
      real, dimension(:), allocatable, target :: a, b
      real, dimension(:), pointer :: ptr1=>NULL(), ptr2=>NULL()
      real :: t

      integer :: i

      allocate(a(100), b(100))

      do i=1, 100
        a(i)=i
        b(i)=100-i
      end do

      ! Test pointing to a target
      ptr1 => a
      do i=1, 100
        call assert(ptr1(i)==real(i), __FILE__, __LINE__)
      end do

      ! Test another pointer pointing to a target
      ptr2 => a
      do i=1, 100
        call assert(ptr2(i)==real(i), __FILE__, __LINE__)
      end do

      ! Test assigning pointer to a pointer
      ptr3 => ptr1
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
  end subroutine calc

  subroutine swap(swp1, swp2)
    real, dimension(:), pointer :: swp1, swp2

      real, dimension(:), pointer :: swp_tmp

      swp_tmp => swp1
      swp1 => swp2
      swp2 => swp_tmp
  end subroutine swap
end module pointer_test

#ifndef FRAGMENT_ONLY
program driver
  use pointer_test
  implicit none

  call assert_init(.false.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
