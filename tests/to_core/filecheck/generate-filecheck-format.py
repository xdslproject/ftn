#!/bin/python3

import sys

in_file = open(sys.argv[1], 'r')
out_file = open(sys.argv[1]+"_test", 'w')

lines = in_file.readlines()

for idx, line in enumerate(lines):  
  if idx == 0:
    out_file.write("//CHECK:       ")
  else:
    if len(line.strip()) == 0:
      out_file.write("//CHECK-EMPTY:  ")
    else:
      out_file.write("//CHECK-NEXT:  ")
  out_file.write(line)
  
in_file.close()
out_file.close()
