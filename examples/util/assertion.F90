module assertion
  implicit none

  integer :: passed_tests=0, failed_tests=0
  logical :: fail_on_error=.true.

  private
  public :: assert, assert_init, assert_finalize

contains

  subroutine assert(condition, file, line)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: file
    integer, intent(in) :: line

    if (.not. condition) then
      print *, "Error in '", file, "' at line ", line
      if (fail_on_error) then
        call exit(-1)
      else
        failed_tests=failed_tests+1
      end if
    else
      passed_tests=passed_tests+1
    end if
  end subroutine assert

  subroutine assert_init(raise_error_on_fail)
    logical, intent(in) :: raise_error_on_fail

    fail_on_error=raise_error_on_fail
  end subroutine assert_init

  subroutine assert_finalize(file)
    character(len=*), intent(in) :: file

    if (failed_tests .gt. 0) then
      print *, "[FAIL] '", file, "' Passes: ", passed_tests, "Fails: ", failed_tests
    else
      print *, "[PASS] '", file, "' Passes: ", passed_tests
    end if
  end subroutine assert_finalize
end module assertion
