#!/usr/bin/env python3.10

from os import listdir, system, environ, remove, rmdir
from os.path import isfile, join, exists
import shutil
import subprocess
import glob
import concurrent.futures


def build_executable(src, exec_name, object_files, include_dir):
    cmd_args = ""
    for of in object_files:
        cmd_args += f" --linkobj {of}"
    for inc in include_dir:
        cmd_args += f" -I{inc}"
    system(f"xftn {src} -o {exec_name} {cmd_args} -v0")
    return exists(exec_name)


def build_objectfile(src, obj_name):
    system(f"xftn {src} -o {obj_name} --flang-module-dir . -v0")


def run_executable(exec_name):
    result = subprocess.run(
        [f"./{exec_name}"], shell=False, capture_output=True, text=True
    )
    output = result.stdout
    out_components = output.split("\n")
    run_success = True
    if len(out_components) > 1:
        for prt_a, prt_b in zip(*[iter(out_components)] * 2):
            # This formats properly as it will reduce the line width and add a newline
            # but does assume there is one newline per line, which is not guaranteed
            # The character after newline is always an additional space, therefore ignore it
            pretty_out = prt_a + prt_b[1:]
            if ":" in pretty_out:
                pretty_out_components = pretty_out.split(":")
                pretty_out = ""
                for idx, comp in enumerate(pretty_out_components):
                    if idx > 0:
                        pretty_out += ": "
                    pretty_out += comp.strip()
            pretty_out = pretty_out.strip()
            print(pretty_out)
            if "FAIL" in pretty_out:
                run_success = False
    else:
        pretty_out = output
        print(pretty_out)
        if "FAIL" in pretty_out:
            run_success = False

    if len(result.stderr.strip()) > 0:
        print(result.stderr.strip())

    if result.returncode != 0:
        run_success = False
        print(f"[ERROR] '{exec_name}' returned code {result.returncode}")

    return run_success


if not "XFTN_PATH" in environ:
    print(
        "Error: 'XFTN_PATH' environment must be set to the root of the 'ftn' repository"
    )
    exit(-1)

examples_dir = join(environ["XFTN_PATH"], "examples")
fragments_dir = join(examples_dir, "fragments")

src_files = [
    f
    for f in listdir(fragments_dir)
    if isfile(join(fragments_dir, f)) and f.endswith(".F90")
]

link_obj_files = ["assertion.o"]
include_dir = []

print("-------------------------------")
print("Building dependencies")
print("-------------------------------")
build_objectfile(join(examples_dir, "util/assertion.F90"), "assertion.o")
print("Built assertion")

print("\n-------------------------------")
print("Building tests")
print("-------------------------------")

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {}
    for src_file in src_files:
        source_fn_no_ext = src_file.split(".")[-2]
        absolute_src_file = join(fragments_dir, src_file)
        future = executor.submit(
            build_executable,
            absolute_src_file,
            source_fn_no_ext,
            link_obj_files,
            include_dir,
        )
        futures[future] = absolute_src_file

    for future in concurrent.futures.as_completed(futures):
        built_executable = future.result()
        if built_executable:
            print(f"Built source file '{futures[future]}'")
        else:
            print(f"Build failed for '{futures[future]}'")
            exit(-1)


print("\n-------------------------------")
print("Running tests")
print("-------------------------------")
failures = []
for src_file in src_files:
    source_fn_no_ext = src_file.split(".")[-2]
    if not run_executable(source_fn_no_ext):
        failures.append(source_fn_no_ext)

# Only clean up if all successful (or don't clean up those that failed!)
print("\n-------------------------------")
print("Cleaning up successful tests")
print("-------------------------------")
for src_file in src_files:
    source_fn_no_ext = src_file.split(".")[-2]
    if source_fn_no_ext not in failures:
        remove(source_fn_no_ext)
        for f in glob.glob(f"tmp/{source_fn_no_ext}*"):
            remove(f)

if len(failures) == 0:
    # Only remove assertion if there are no failues, as otherwise might
    # want to reuse when recompiling for testing
    remove("assertion.o")
    remove("assertion.mod")
    for f in glob.glob(f"tmp/assertion*"):
        remove(f)

if len(listdir("tmp")) == 0:
    rmdir("tmp")

print("Done")

print("\n===============================")
if len(failures) > 0:
    print(
        f"{len(src_files) - len(failures)} tests passed, {len(failures)} tests failed"
    )
else:
    print("All tests passed")
print("===============================")
