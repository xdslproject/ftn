# Filecheck testing

## Running tests

To run the tests `./run_filecheck.sh` , you will need to set up the environment first so it can find `xftn`, `flang` etc.

## Regenerating MLIR via automated script

If the transformation changes then the comparison MLIR will need to be regenerated, you can do this more manually (see next section) or with the `update-filecheck-test` script that will regenerate the test file with the latest output from xftn. 

For example, for `assertion.test`

```
python3 update-filecheck-test.py util/assertion.test ../../../examples/util/assertion.F90
```

This is the quickest way to update each test when the MLIR changes, and is the suggested approach. It automates the manual steps in the next section

## Regenerating MLIR slightly more manually

If the transformation changes then the comparison MLIR will need to be regenerated, check the arguments in the test (e.g. it might have `--openmp` or `--offload`). The only difference is that the tests generate to stdout for piping, whereas we need to generate to a file for processing. Then run something like:

```
xftn ../../../examples/solvers/jacobi.F90 --openmp --cleanup --stages=flang,pre,ftn -v0 -o out.mlir
python3 generate-filecheck-format.py out.mlir
```

or (without OpenMP)

```
xftn ../../../examples/solvers/jacobi.F90 --cleanup --stages=flang,pre,ftn -v0 -o out.mlir
python3 generate-filecheck-format.py out.mlir
```

of (for OpenMP target offload)

```
xftn ../../../examples/omp/offload/ex1.F90 --cleanup --offload -v0 -o out.mlir
python3 generate-filecheck-format.py out.mlir
```

This will generate an output file that you can copy and paste into the test. Note that you need to be careful with `--openmp` as even if there is no OpenMP in the code if the flag is enabled it will generate some preamble, so the test command and this generation must be consistent.
