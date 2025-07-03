module problem_mod	
  implicit none	

  ! Boundary value at the LHS of the pipe
  real(kind=8), parameter :: LEFT_VALUE = 1.0
  ! Boundary value at the RHS of the pipe
  real(kind=8), parameter :: RIGHT_VALUE = 10.0
  ! The maximum number of iterations
  integer, parameter :: MAX_ITERATIONS = 100000
  ! How often to report the norm
  integer, parameter :: REPORT_NORM_PERIOD = 100

contains

  subroutine run_solver(nx, ny, convergence_accuracy)
    integer, intent(in) :: nx, ny
    real(kind=8),  intent(in) :: convergence_accuracy
    
    real(kind=8), dimension(:,:), allocatable :: u_k, u_kp1, temp
    real(kind=8) :: bnorm, rnorm, norm
    integer :: i, j, k

    print *, "Global size in X=", nx, "Global size in Y=", ny

    allocate(u_k(0:ny+1, 0:nx+1), u_kp1(0:ny+1, 0:nx+1))!, temp(0:ny+1, 0:nx+1))

    bnorm=0.0
    rnorm=0.0
    
    call initialise_values(u_k, u_kp1, nx, ny)

    ! Calculate the initial residual
    !$omp parallel do reduction(+:bnorm)
    do i=1, nx
      do j=1, ny
        bnorm=bnorm+((u_k(j,i)*4-u_k(j-1,i)-u_k(j+1,i)-u_k(j,i-1)-u_k(j,i+1)) ** 2)
      end do
    end do
    !$omp end parallel do
    ! In the parallel version you will be operating on only part of the domain in each process, so you will need to do some
    ! form of reduction to determine the global bnorm before square rooting it
    bnorm=sqrt(bnorm)

    do k=0, MAX_ITERATIONS      
      ! The halo swapping will likely need to go in here
      rnorm=0.0
      ! Calculates the current residual norm
      !$omp parallel do reduction(+:rnorm)
      do i=1, nx
        do j=1, ny
          rnorm=rnorm+((u_k(j,i)*4-u_k(j-1,i)-u_k(j+1,i)-u_k(j,i-1)-u_k(j,i+1)) ** 2)
        end do
      end do
      !$omp end parallel do
      ! In the parallel version you will be operating on only part of the domain in each process, so you will need to do some
      !  form of reduction to determine the global rnorm before square rooting it
      norm=sqrt(rnorm)/bnorm

      if (norm .lt. convergence_accuracy) exit

      ! Do the Jacobi iteration
      !$omp parallel do
      do i=1, nx
        do j=1, ny
          u_kp1(j,i)=0.25 * (u_k(j-1,i) + u_k(j+1,i) + u_k(j,i-1) + u_k(j,i+1))
        end do
      end do
      !$omp end parallel do

      ! Swap data structures round for the next iteration
      call move_alloc(u_kp1, temp)
      call move_alloc(u_k, u_kp1)
      call move_alloc(temp, u_k)
      
      if (mod(k, REPORT_NORM_PERIOD)==0) print *, "Iteration=",k," Relative Norm=",norm
    end do
    print *, "Terminated on ",k," iterations, Relative Norm=", norm
    deallocate(u_k, u_kp1)
  end subroutine run_solver

  ! Initialises the arrays, such that u_k contains the boundary conditions at the start and end points and all other
  ! points are zero. u_kp1 is set to equal u_k
  subroutine initialise_values(u_k, u_kp1, nx, ny)
    real(kind=8), intent(inout) :: u_k(0:ny+1, 0:nx+1), u_kp1(0:ny+1, 0:nx+1)
    integer, intent(in) :: nx, ny
    
    integer :: i, j

    ! We are setting the boundary (left and right) values here, in the parallel version this should be exactly the same and no changed required

    do i=0, nx+1
      do j=1, ny
        u_k(j,i)=0.0_8				
      end do
    end do
    
    do i=0, nx+1
      u_k(0,i)=LEFT_VALUE
      u_k(ny+1,i)=RIGHT_VALUE
    end do		
    u_kp1=u_k
  end subroutine initialise_values  
end module problem_mod

program diffusion
  use problem_mod
  implicit none

  call run_solver(512, 512, 1D-4)
end program diffusion
