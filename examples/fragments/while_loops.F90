#ifndef FRAGMENT_ONLY
module while_loops_test
  use assertion
  implicit none

contains
#endif

  subroutine calc(a)
    integer, intent(in) :: a

    integer :: n, c

    ! Test counting upwards
    c=0
    n=1
    do while (n .le. a)
      c=c+n
      n=n+1
    end do
    call assert(c==5050, __FILE__, __LINE__)

    ! Test counting upwards with a step
    c=0
    n=1
    do while (n .le. a)
      c=c+n
      n=n+2
    end do
    call assert(c==2500, __FILE__, __LINE__)

    ! Test counting downwards
    c=10000
    n=a
    do while (n .ge. 1)
      c=c-n
      n=n-1
    end do
    call assert(c==4950, __FILE__, __LINE__)

    ! Test counting downwards with a step
    c=10000
    n=a
    do while (n .ge. 1)
      c=c-n
      n=n-15
    end do
    call assert(c==9615, __FILE__, __LINE__)

    ! Test exiting from a loop
    c=0
    n=1
    do while (n .le. a)
      c=c+n
      if (n .gt. 10) exit
      n=n+1
    end do
    call assert(c==66, __FILE__, __LINE__)

    ! Test loop cycles
    c=0
    n=1
    do while (n .le. a)
      n=n+1
      if (n-1 .gt. 20) cycle
      c=c+(n-1)
    end do
    call assert(c==210, __FILE__, __LINE__)
  end subroutine calc

#ifndef FRAGMENT_ONLY
end module while_loops_test

program driver
  use while_loops_test
  implicit none

  call assert_init(.false.)
  call calc(100)
  call assert_finalize(__FILE__)
end program driver
#endif
