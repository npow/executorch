#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TOOLS=/Users/npow/.vcpkg/artifacts/2139c4c6
export CMSIS_PACK_ROOT=/Users/npow/.cache/arm/packs
export GCC_TOOLCHAIN_15_3_1="$TOOLS/compilers.arm.arm.none.eabi.gcc/15.3.1/bin"
export PATH="$TOOLS/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/bin:$TOOLS/tools.kitware.cmake/4.3.3/bin:$TOOLS/tools.ninja.build.ninja/1.13.2:$GCC_TOOLCHAIN_15_3_1:$PATH"
exec cbuild "$ROOT/firmware/cmsis-executorch.csolution.yml" --active DevKit-E8 "$@"
