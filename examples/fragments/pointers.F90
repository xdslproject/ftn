subroutine calc()
    real, dimension(:), allocatable, target :: a
    real, dimension(:), pointer :: ptr1=>NULL(), ptr2=>NULL()
    real :: t

    allocate(a(100))

    ptr1 => a
    ptr2 = ptr1

    ptr1(20)=34

    ptr1 => ptr2

    t=ptr1(3)
    ptr1 => NULL()
end subroutine calc

! todo - pass pointers to subroutine and manipulate
