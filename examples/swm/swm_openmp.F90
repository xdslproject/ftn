module swm_openmp_mod
  implicit none

  logical, parameter :: L_OUT = .true.

contains

  subroutine run_model(M, N, M_LEN, N_LEN, ITMAX)
    integer, intent(in) :: M, N, M_LEN, N_LEN, ITMAX
    ! Solution arrays
    real, dimension(:,:), allocatable :: u, unew, uold, &
                                            v, vnew, vold, &
                                            p, pnew, pold, temp

    real, dimension(:,:), allocatable :: cu, &
                                    cv, &
                                    z, &
                                    h, &
                                    psi


    real :: dt, tdt, dx, dy, a, alpha, el, pi
    real :: tpi, di, dj, pcf
    real :: tdts8, tdtsdx, tdtsdy, fsdx, fsdy

    integer :: mnmin
    integer :: ncycle
    integer :: i,j

    ! Timer variables
    real :: mfs100, mfs200, mfs300
    real :: t100, t200, t300
    real :: tstart, ctime, tcyc, time, ptime
    real :: c1, c2

    mnmin = min(M,N)

    allocate(u(M_LEN, N_LEN), unew(M_LEN, N_LEN), uold(M_LEN, N_LEN))
    allocate(v(M_LEN, N_LEN), vnew(M_LEN, N_LEN), vold(M_LEN, N_LEN))
    allocate(p(M_LEN, N_LEN), pnew(M_LEN, N_LEN), pold(M_LEN, N_LEN))

    allocate(cu(M_LEN, N_LEN), cv(M_LEN, N_LEN), z(M_LEN, N_LEN), h(M_LEN, N_LEN), psi(M_LEN, N_LEN))

    ! Initialization
    dt = 90.
    tdt = dt

    dx = 1.e5
    dy = 1.e5
    fsdx = 4. / dx
    fsdy = 4. / dy

    a = 1.e6
    alpha = 0.001

    el = N * dx
    pi = atan2(0., -1.)
    tpi = pi + pi
    di = tpi / M
    dj = tpi / N
    pcf = pi * pi * a * a / (el * el)

    ! Initial values of the stream function and p
    do j=0,N_LEN-1
      do i=0,M_LEN-1
        psi(i+1,j+1) = a * sin((i + .5) * di) * sin((j + .5) * dj)
        p(i+1,j+1) = pcf * (cos(2. * (i) * di) + cos(2. * (j) * dj)) + 50000.
      end do
    end do

    ! Initialize velocities
    do j=1,N
      do i=1,M
        u(i+1,j) = -(psi(i+1,j+1) - psi(i+1,j)) / dy
        v(i,j+1) = (psi(i+1,j+1) - psi(i,j+1)) / dx
      end do
    end do

    ! Periodic continuation
    do j=1,N
      u(1,j) = u(M_LEN,j)
      v(M_LEN,j) = v(1,j)
    end do

    do i=1,M
      u(i,N_LEN) = u(i,1)
      v(i,1) = v(i,N_LEN)
    end do

    u(1,N_LEN) = u(M_LEN,1)
    v(M_LEN,1) = v(1,N_LEN)

    do j=1,N_LEN
      do i=1,M_LEN
        uold(i,j) = u(i,j)
        vold(i,j) = v(i,j)
        pold(i,j) = p(i,j)
      end do
    end do

    if ( L_OUT ) then
      write(*,"(A,I0)") " number of points in the x direction ", N
      write(*,"(A,I0)") " number of points in the y direction ", M
      write(*,"(A,F0.6)") " grid spacing in the x direction     ", dx
      write(*,"(A,F0.6)") " grid spacing in the y direction     ", dy
      write(*,"(A,F0.6)") " time step                           ", dt
      write(*,"(A,F0.6)") " time filter parameter               ", alpha

      write(*, "(A)") " initial diagonal elements of p"
      do i=1,mnmin
        write(*, "(F0.6, 1X)", advance="no") p(i,i)
      end do

      write(*, "(/,A)") " initial diagonal elements of u"
      do i=1,mnmin
        write(*, "(F0.6, 1X)", advance="no") u(i,i)
      end do

      write(*, "(/,A)") " initial diagonal elements of v"
      do i=1,mnmin
        write(*, "(F0.6, 1X)", advance="no") v(i,i)
      end do
      write(*,*)
    end if

    ! Copy data in
    !$omp target enter data map(to: fsdx,fsdy,alpha,p,u,v,pold,uold,vold) map(alloc: cu,cv,h,z,pnew,unew,vnew,tdts8,tdtsdx,tdtsdy)

    ! Start timer
    call cpu_time(tstart)
    time = 0.
    t100 = 0.
    t200 = 0.
    t300 = 0.

    ! Start of time loop
    do ncycle=1,ITMAX

      call cpu_time(c1)
      !$omp target parallel do collapse(2)
      do j=1,N
        do i=1,M
          cu(i+1,j) = 0.5 * (p(i+1,j) + p(i,j)) * u(i+1,j)
          cv(i,j+1) = 0.5 * (p(i,j+1) + p(i,j)) * v(i,j+1)
          z(i+1,j+1) = (fsdx * (v(i+1,j+1) - v(i,j+1)) - fsdy * (u(i+1,j+1) - u(i+1,j))) / &
                       (p(i,j) + p(i+1,j) + p(i+1,j+1) + p(i, j+1))
          h(i,j) = p(i,j) + 0.25 * (u(i+1,j) * u(i+1,j) + u(i,j) * u(i,j) + &
                                    v(i,j+1) * v(i,j+1) + v(i,j) * v(i,j))
        end do
      end do
      !$omp end target parallel do
      call cpu_time(c2)
      t100 = t100 + (c2 - c1)

      ! Periodic continuation
      !$omp target
      !$omp parallel do
      do j=1,N
        cu(1,j) = cu(M_LEN,j)
        cv(M_LEN,j+1) = cv(1,j+1)
        z(1,j+1) = z(M_LEN,j+1)
        h(M_LEN,j) = h(1,j)
      end do
      !$omp end parallel do

      !$omp parallel do
      do i=1,M
        cu(i+1, N_LEN) = cu(i+1,1)
        cv(i,1) = cv(i,N_LEN)
        z(i+1,1) = z(i+1,N_LEN)
        h(i,N_LEN) = h(i,1)
      end do
      !$omp end parallel do

      cu(1,N_LEN) = cu(M_LEN,1)
      cv(M_LEN,1) = cv(1,N_LEN)
      z(1,1) = z(M_LEN,N_LEN)
      h(M_LEN,N_LEN) = h(1,1)

      !$omp end target

      ! Compute new values of u, v, and p
      tdts8 = tdt / 8.
      tdtsdx = tdt / dx
      tdtsdy = tdt / dy

      call cpu_time(c1)
      !$omp target parallel do collapse(2)
      do j=1,N
        do i=1,M
          unew(i+1,j) = uold(i+1,j) + &
                        tdts8 * (z(i+1,j+1) + z(i+1,j)) * (cv(i+1,j+1) + cv(i,j+1) + cv(i,j) + cv(i+1,j)) - &
                        tdtsdx * (h(i+1,j) - h(i,j))
          vnew(i,j+1) = vold(i,j+1) - &
                        tdts8 * (z(i+1,j+1) + z(i,j+1)) * (cu(i+1,j+1) + cu(i,j+1) + cu(i,j) + cu(i+1,j)) - &
                        tdtsdy * (h(i,j+1) - h(i,j))
          pnew(i,j) = pold(i,j) - tdtsdx * (cu(i+1,j) - cu(i,j)) - tdtsdy * (cv(i,j+1) - cv(i,j))
        end do
      end do
      !$omp end target parallel do
      call cpu_time(c2)
      t200 = t200 + (c2-c1)

      ! Periodic continuation
      !$omp target
      !$omp parallel do
      do j=1,N
        unew(1,j) = unew(M_LEN,j)
        vnew(M_LEN,j+1) = vnew(1,j+1)
        pnew(M_LEN,j) = pnew(1,j)
      end do
      !$omp end parallel do

      !$omp parallel do
      do i=1,M
        unew(i+1,N_LEN) = unew(i+1,1)
        vnew(i,1) = vnew(i,N_LEN)
        pnew(i,N_LEN) = pnew(i,1)
      end do
      !$omp end parallel do

      unew(1,N_LEN) = unew(M_LEN,1)
      vnew(M_LEN,1) = vnew(1,N_LEN)
      pnew(M_LEN,N_LEN) = pnew(1,1)
      !$omp end target

      time = time + dt
      if (ncycle > 1) then
        call cpu_time(c1)
        !$omp target parallel do collapse(2)
        do j=1,N_LEN
          do i=1,M_LEN
            uold(i,j) = u(i,j) + alpha*(unew(i,j) - 2. * u(i,j) + uold(i,j))
            vold(i,j) = v(i,j) + alpha*(vnew(i,j) - 2. * v(i,j) + vold(i,j))
            pold(i,j) = p(i,j) + alpha*(pnew(i,j) - 2. * p(i,j) + pold(i,j))
          end do
        end do
        !$omp end target parallel do

        call move_alloc(u, temp)
        call move_alloc(unew, u)
        call move_alloc(temp, unew)

        call move_alloc(v, temp)
        call move_alloc(vnew, v)
        call move_alloc(temp, vnew)

        call move_alloc(p, temp)
        call move_alloc(pnew, p)
        call move_alloc(temp, pnew)

        call cpu_time(c2)
        t300 = t300 + (c2 - c1)
      else ! ncycle = 1
        tdt = tdt + tdt
        !$omp target parallel do
        do j=1,N_LEN
          do i=1,N_LEN
            uold(i,j) = u(i,j)
            vold(i,j) = v(i,j)
            pold(i,j) = p(i,j)
          end do
        end do
        !$omp end target parallel do
        call move_alloc(u, temp)
        call move_alloc(unew, u)
        call move_alloc(temp, unew)

        call move_alloc(v, temp)
        call move_alloc(vnew, v)
        call move_alloc(temp, vnew)

        call move_alloc(p, temp)
        call move_alloc(pnew, p)
        call move_alloc(temp, pnew)
      end if
    end do ! End of time loop

    !$omp target exit data map(from: cu,cv,h,z,pnew,unew,vnew,p,u,v,uold,vold,pold)

    call move_alloc(u, temp)
    call move_alloc(unew, u)
    call move_alloc(temp, unew)

    call move_alloc(v, temp)
    call move_alloc(vnew, v)
    call move_alloc(temp, vnew)

    call move_alloc(p, temp)
    call move_alloc(pnew, p)
    call move_alloc(temp, pnew)

    if ( L_OUT ) then
      ptime = time / 3600.

      write(*, "(A,I0,A,F0.6)") " cycle number ", ITMAX, &
                                " model time in hours ", ptime
      write(*, "(A)") " diagonal elements of p"
      do i=1,mnmin
        write(*, "(F0.6,1X)", advance="no") pnew(i,i)
      end do
      write(*, "(/,A)") " diagonal elements of u"
      do i=1,mnmin
        write(*, "(F0.6,1X)", advance="no") unew(i,i)
      end do
      write(*, "(/,A)") " diagonal elements of v"
      do i=1,mnmin
        write(*, "(F0.6,1X)", advance="no") vnew(i,i)
      end do

      mfs100 = 0.
      mfs200 = 0.
      mfs300 = 0.
      ! gdr t100 etc. now an accumulation of all l100 time
      if ( t100 .gt. 0 ) mfs100 = real(ITMAX) * 24. * real(M*N) / t100 / 1000000.
      if ( t200 .gt. 0 ) mfs200 = real(ITMAX) * 26. * real(M*N) / t200 / 1000000.
      if ( t300 .gt. 0 ) mfs300 = real(ITMAX) * 15. * real(M*N) / t300 / 1000000.

      call cpu_time(c2)
      ctime = c2 - tstart
      tcyc = ctime / real(ITMAX)

      write(*, "(/,A,I0,A,F0.6,A,F0.6)") " cycle number ", ITMAX, " total computer time ", ctime, " time per cycle ", tcyc
      write(*, "(A,F0.6,1X,F0.6)") " time and megaflops for loop 100 ", t100, mfs100
      write(*, "(A,F0.6,1X,F0.6)") " time and megaflops for loop 200 ", t200, mfs200
      write(*, "(A,F0.6,1X,F0.6)") " time and megaflops for loop 300 ", t300, mfs300
    end if

    deallocate(u, unew, uold, v, vnew, vold, p, pnew, pold, cu, cv, z, h, psi)
  end subroutine run_model
end module swm_openmp_mod

program swm
  use swm_openmp_mod

implicit none

  integer, parameter :: M = 512
  integer, parameter :: N = 512
  integer, parameter :: M_LEN = M + 1
  integer, parameter :: N_LEN = N + 1
  integer, parameter :: ITMAX = 4000

  call run_model(M, N, M_LEN, N_LEN, ITMAX)
end program swm
