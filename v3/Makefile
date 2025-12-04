##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.txt for details.
#
# (C) 2024-2025 Advanced Micro Devices, Inc. or its affiliates
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common

# ---------------- Dispositivo / target ----------------
targetname ?= spmv_paged_design

# Si no se define NPU2, por defecto usamos npu2
NPU2 ?= 1
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

# ---------------- Parámetros del problema ----------------
ROWS       ?= 8         # M
COLS       ?= 8         # N
NNZ        ?= 8         # solo para info / futuro, el diseño pagina A igual

# Paginado
RPB        ?= 8         # rows per block (page_rows_max)
CHUNK_NNZ  ?= 8         # page_nnz_max
N_AIE_COLS ?= 1         # cuántas columnas de AIE usamos

# Tipos
DTYPE_IN   ?= i32
DTYPE_OUT  ?= i32

# ---------------- Buffers / alloc scheme ----------------
use_linear_buf_alloc ?= 1
buffer_aloc_flag := $(if $(filter 1,$(use_linear_buf_alloc)),basic-sequential,bank-aware)

# ---------------- Fuentes ----------------
aie_py_src ?= vibecodeado.py
aie_core_cc?= spmv.cc          # kernel AIE (spmv_cascade_... o el que uses)
host_src   ?= test.cpp

# Nombre del ejecutable host
host_out := ${targetname}.exe

# ---------------- Kernels AIE ----------------
kernels := spmv_sparse         # nombre lógico del .o que linkea el core

# Compilador para los kernels AIE
KERNEL_CC := ${PEANO_INSTALL_DIR}/bin/clang++

# Flags según device
ifeq ($(devicename),npu2)
  KERNEL_CFLAGS := ${PEANOWRAP2P_FLAGS}
else
  KERNEL_CFLAGS := ${PEANOWRAP2_FLAGS}
endif

# Defines que usan tus kernels C++ (DIM_RPB / DIM_CHUNK_NNZ)
KERNEL_DEFINES := -DDIM_RPB=${RPB} -DDIM_CHUNK_NNZ=${CHUNK_NNZ}

# ---------------- aiecc / peano ----------------
aiecc_peano_flags = --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}

# ---------------- Args para el script Python ----------------
aieargs += --dev ${devicename} -R ${ROWS} -C ${COLS}
aieargs += --nnz ${NNZ} --page-rows-max ${RPB} --page-nnz-max ${CHUNK_NNZ}
aieargs += --n-aie-cols ${N_AIE_COLS} --dtype_in ${DTYPE_IN} --dtype_out ${DTYPE_OUT}
aieargs += --trace_size 0

# ---------------- Targets de artefactos ----------------
mlir_target        ?= build/aie.mlir
xclbin_target      ?= build/final.xclbin
insts_elf_target   ?= build/insts.elf   # este es el que usa el host (-i)
insts_txt_target   ?= build/insts.txt   # npu insts (no lo usa el host pero lo generamos)

powershell ?=
getwslpath ?=
runargs ?= -v 2 --verify 1

.PHONY: all run clean print-vars

# ======================================================================
#  Target alto nivel
# ======================================================================
all: ${xclbin_target} ${host_out}

# ======================================================================
#  Kernels AIE (.o)
# ======================================================================
# build/spmv_sparse.o ← spmv.cc
build/spmv_sparse.o: ${srcdir}/${aie_core_cc}
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

# ======================================================================
#  MLIR (graph / device)
# ======================================================================
${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} > $@

# ======================================================================
#  XCLBIN + insts.* (aiecc.py)
# ======================================================================
${xclbin_target}: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py \
		--alloc-scheme=${buffer_aloc_flag} \
		--aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		${aiecc_peano_flags} \
		--aie-generate-npu-insts --npu-insts-name=$(notdir ${insts_txt_target}) \
		--aie-generate-elf --elf-name=$(notdir ${insts_elf_target}) \
		$(^:%=../%)

# ======================================================================
#  Host (CMake)
# ======================================================================
${host_out}: ${srcdir}/${host_src}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

# ======================================================================
#  Run
# ======================================================================
run: ${host_out} ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./${host_out} \
		-x ${xclbin_target} \
		-i ${insts_elf_target} \
		-k MLIR_AIE \
		${runargs}

# ======================================================================
#  Utils
# ======================================================================
print-vars:
	@echo devicename=$(devicename)
	@echo ROWS=$(ROWS) COLS=$(COLS) NNZ=$(NNZ)
	@echo RPB=$(RPB) CHUNK_NNZ=$(CHUNK_NNZ) N_AIE_COLS=$(N_AIE_COLS)
	@echo DTYPE_IN=$(DTYPE_IN) DTYPE_OUT=$(DTYPE_OUT)
	@echo aie_py_src=$(aie_py_src)
	@echo aie_core_cc=$(aie_core_cc)
	@echo host_src=$(host_src)
	@echo kernels=$(kernels)
	@echo mlir_target=$(mlir_target)
	@echo xclbin_target=$(xclbin_target)
	@echo insts_elf_target=$(insts_elf_target)

clean:
	rm -rf build _build ${host_out}
