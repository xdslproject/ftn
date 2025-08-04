#!/bin/bash

DOCKER_RUN="docker run -i -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd)"

alias python-soda="$DOCKER_RUN agostini01/soda:v15.08:v15.08 python3"
alias soda-opt="$DOCKER_RUN agostini01/soda:v15.08 soda-opt"
alias opt-16="$DOCKER_RUN agostini01/soda:v15.08 opt"
alias mlir-opt="$DOCKER_RUN agostini01/soda:v15.08 mlir-opt"
alias soda-mlir-translate="$DOCKER_RUN agostini01/soda:v15.08 mlir-translate"