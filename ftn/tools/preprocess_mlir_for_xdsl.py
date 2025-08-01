#!/usr/bin/env python3.10

import sys
import struct
from typing import IO


def preproc(file: IO[str]) -> str:
  lines=""
  for line in file:
    #line=line.replace(", omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>", "")
    if "arith.constant" in line and "() -> f64" in line and "value = 0x" in line:
      # This is the hex value that we need to convert
      hex_val=line.split("value =")[1].split(":")[0].strip()
      rhex_val=hex_val.replace("0x", "")
      float_val=None
      try:
        # Try to convert to a double
        float_val=struct.unpack('!d', bytes.fromhex(rhex_val))[0]
      except:
        # Try to convert to a single float
        float_val=struct.unpack('!f', bytes.fromhex(rhex_val))[0]

      if float_val is not None:
        to_add_line=line.split("value =")[0]+"value ="+str(float_val)+" : "+line.split("value =")[1].split(":", 1)[1]
      else:
        print("Warning, could not convert hex value '{hex_val}' to float, defaulting to existing value", file=sys.stderr)
        to_add_line=line
    if "\\00" in line:
      # Ensure we don't have null terminator at the end
      line=line.replace("\\00", "")
      t1=line.split("size = ")
      on_num=t1[1].split(" :", 1)
      new_len=int(on_num[0])-1

      to_add_line=t1[0]+"size = "+str(new_len)+" :"+on_num[1]
    else:
      to_add_line=line
    lines+=to_add_line

  return lines

def main():
    assert len(sys.argv) == 3

    with open(sys.argv[1]) as file:
        lines = preproc(file)

    with open(sys.argv[2], "w") as file:
      file.write(lines)


if __name__ == "__main__":
    main()
