module proc_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

  integer :: v1
  integer :: v2=20
  integer, parameter :: c_v = 100

contains

  subroutine calc()
    integer :: val

    ! Tests global scalars
    call assert(v2 == 20, __FILE__, __LINE__)
    call assert(c_v == 100, __FILE__, __LINE__)

    v1=13
    call assert(v1 == 13, __FILE__, __LINE__)
    v2=87
    call assert(v2 == 87, __FILE__, __LINE__)

    ! Ensure they are actually global, update in procedure reflected here
    call mod_globals()
    call assert(v1 == 99, __FILE__, __LINE__)
    call assert(v2 == 66, __FILE__, __LINE__)

    ! Now test a permutation of call subroutines and functions
    call proc_one(5, val)
    call assert(val == 50, __FILE__, __LINE__)
    call proc_two(9, val)
    call assert(val == 900, __FILE__, __LINE__)
    val=fn1(1)
    call assert(val == 10, __FILE__, __LINE__)
    val=fn2(2)
    call assert(val == 200, __FILE__, __LINE__)
    val=fn3(3)
    call assert(val == 3000, __FILE__, __LINE__)
    val=fn4(4)
    call assert(val == 40000, __FILE__, __LINE__)
  end subroutine calc

  subroutine proc_one(a, b)
    integer, intent(in) :: a
    integer, intent(out) :: b

    b = a * 10
  end subroutine proc_one

  subroutine proc_two(a, b)
    integer, intent(in) :: a
    integer, intent(inout) :: b

    b = a * 100

    ! Test return from subroutine
    return
    call assert(.false., __FILE__, __LINE__)
  end subroutine proc_two

  function fn1(v) result (res)
    integer, intent (in) :: v
    integer :: res

    res = v * 10
  end function fn1

  integer function fn2(v) result (res)
    integer, intent (in) :: v

    res = v * 100
  end function fn2

  integer function fn3(v)
    integer, intent(in) :: v

    fn3 = v * 1000
  end function fn3

  function fn4(v)
    integer, intent (in) :: v
    integer :: fn4

    fn4 = v * 10000

    ! Test return from function
    return
    call assert(.false., __FILE__, __LINE__)
  end function fn4

  subroutine mod_globals()
    v1=99
    v2=66
  end subroutine mod_globals
end module proc_test

#ifndef FRAGMENT_ONLY
program driver
  use proc_test
  implicit none

  call assert_init(.true.)
  call calc()
  call assert_finalize(__FILE__)
end program driver
#endif
