module array_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

      real :: global_array(100)

contains

  subroutine calc()
      real, dimension(100) :: a
      real :: b(100), t

      integer :: i

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
end module array_test

#ifndef FRAGMENT_ONLY
program driver
  use array_test
  implicit none

  call assert_init(.true.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
