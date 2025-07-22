#ifndef FRAGMENT_ONLY
module mod_example
  use assertion
  implicit none

contains
#endif

  subroutine example(a, b)
    integer, intent(in) :: a, b

    integer :: c, d

    if (a == 100) then
      c=23
      d=2
    else
      c=82
      d=1
    end if

    if (b == 200 .and. c == 23) then
      call assert(d == 2 .and. c == 23, __FILE__, __LINE__)
    end if

    if (a > 99 .and. a < 101) then
      call assert(.true., __FILE__, __LINE__)
    else
      call assert(.false., __FILE__, __LINE__)
    end if

    if (a .ne. 100) then
      call assert(.false., __FILE__, __LINE__)
    end if

    if (a .ne. 100 .or. b .ne. 200) then
      call assert(.false., __FILE__, __LINE__)
    end if

    if (a == 100 .or. b == 200) then
      call assert(.true., __FILE__, __LINE__)
    else
      call assert(.false., __FILE__, __LINE__)
    end if

  end subroutine example

#ifndef FRAGMENT_ONLY
end module mod_example

program driver
  use mod_example
  implicit none

  call assert_init(.false.)
  call example(100, 200)
  call assert_finalize(__FILE__)
end program driver
#endif
