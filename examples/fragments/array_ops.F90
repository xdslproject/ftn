module array_ops_test
#ifndef FRAGMENT_ONLY
  use assertion
#endif
  implicit none

      real :: global_array(10,6)

contains

  subroutine calc(j)
      integer, intent(in) :: j
      real, dimension(100) :: a, b, c
      real, dimension(:,:), allocatable :: n, m
      real, dimension(10, 6) :: p, q

      integer :: i, k

      allocate(n(10,6), m(10,6))

      ! Results in an elemental as do not know what the value of j is
      a=(/ (real(i+j), i = 1, 100) /)
      ! Assign scalar to all elements in array b
      b=20.0
      ! Element wise addition
      c=a+b
      do i=1, 100
        call assert(a(i)==real(i+j), __FILE__, __LINE__)
        call assert(b(i)==20.0, __FILE__, __LINE__)
        call assert(c(i)==a(i) + b(i), __FILE__, __LINE__)
      end do

      ! Tests adding a constant during element wise addition
      c=a+b+100
      do i=1, 100
        call assert(c(i)==a(i) + b(i) + 100, __FILE__, __LINE__)
      end do

      ! Multi-dimension scalar assignment to each element
      n=100.0
      p=100.0
      ! Multi-dimension element wise binary operation
      m=n-p
      do i=1, 10
        do k=1, 6
          call assert(m(i, k)==0.0, __FILE__, __LINE__)
        end do
      end do

      do i=1, 10
        do k=1, 6
          m(i,k)=i
          p(i,k)=k
        end do
      end do

      ! Multi-dumension element wise binary operation across
      ! allocatables and known sizes
      q=m*p
      n=m-p
      do i=1, 10
        do k=1, 6
          call assert(q(i, k)==i*k, __FILE__, __LINE__)
          call assert(n(i, k)==i-k, __FILE__, __LINE__)
        end do
      end do

      ! Global array element wise binary operation
      global_array=m*p*10+n
      do i=1, 10
        do k=1, 6
          call assert(global_array(i, k)==i*k*10+(i-k), __FILE__, __LINE__)
        end do
      end do
  end subroutine calc
end module array_ops_test

#ifndef FRAGMENT_ONLY
program driver
  use array_ops_test
  implicit none

  call assert_init(.false.)
  call calc(10)
  call assert_finalize(__FILE__)
end program driver
#endif
