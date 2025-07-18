#!/usr/bin/env python3.10

import sys
from typing import IO


def postproc(file: IO[str]) -> str:
  lines=""
  for line in file:
    if "arith.constant " in line and (": f64" in line or ": f32" in line):
      to_add_line=line
      ssa_part=line.split("arith.constant ")[0]
      colon_components=line.split("arith.constant ")[1].split(" :")
      numeric_val=colon_components[0]
      if ("e") in numeric_val:
        components=numeric_val.split("e")
        first_part=components[0]
        if (".") not in first_part:
          first_part+=".0"
          numeric_val=first_part+"e"+components[1]
          to_add_line=ssa_part+"arith.constant "+numeric_val+" :"+colon_components[1]
    else:
      to_add_line=line
    lines+=to_add_line
  return lines

def main():
    assert len(sys.argv) == 3

    with open(sys.argv[1]) as file:
        lines = postproc(file)

    with open(sys.argv[2], "w") as file:
      file.write(lines)


if __name__ == "__main__":
    main()
