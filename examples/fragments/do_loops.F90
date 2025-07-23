#ifndef FRAGMENT_ONLY
module mod_example
  use assertion
  implicit none

contains
#endif

  subroutine example(a)
    integer, intent(in) :: a

    integer :: n, c

    ! Test counting upwards
    c=0
    do n = 1, a
      c=c+n
    end do
    call assert(c==5050, __FILE__, __LINE__)

    ! Test counting upwards with a step
    c=0
    do n = 1, a, 2
      c=c+n
    end do
    call assert(c==2500, __FILE__, __LINE__)

    ! Test counting upwards from a specific start point
    c=0
    do n = 80, a
      c=c+n
    end do
    call assert(c==1890, __FILE__, __LINE__)

    ! Test counting downwards
    c=10000
    do n = a, 1, -1
      c=c-n
    end do
    call assert(c==4950, __FILE__, __LINE__)

    ! Test counting downwards with a step
    c=10000
    do n = a, 1, -15
      c=c-n
    end do
    call assert(c==9615, __FILE__, __LINE__)

    ! Test counting downwards from a specific start point
    c=10000
    do n = a-80, 1, -1
      c=c-n
    end do
    call assert(c==9790, __FILE__, __LINE__)

    ! Test exiting from a loop
    c=0
    do n = 1, a
      c=c+n
      if (n .gt. 10) exit
    end do
    call assert(c==66, __FILE__, __LINE__)

    ! Test loop cycles
    c=0
    do n = 1, a
      if (n .gt. 20) cycle
      c=c+n
    end do
    call assert(c==210, __FILE__, __LINE__)

  end subroutine example

#ifndef FRAGMENT_ONLY
end module mod_example

program driver
  use mod_example
  implicit none

  call assert_init(.false.)
  call example(100)
  call assert_finalize(__FILE__)
end program driver
#endif
