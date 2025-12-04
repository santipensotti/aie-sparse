# test_spmv_tile.py
from __future__ import annotations
import numpy as np
import pyxrt as xrt
import sys

import aie.utils.xrt as xrt_utils
import aie.utils.test as test_utils


def _validate_csr_tile(rowptr: np.ndarray, colidx: np.ndarray, tile_rows: int) -> None:
    """Comprueba invariante básicos de CSR: monotonía en rowptr y rangos de índices."""

    if rowptr.ndim != 1:
        raise ValueError("rowptr debe ser un vector 1D")
    if rowptr.size != tile_rows + 1:
        raise ValueError(
            f"rowptr debe tener longitud {tile_rows + 1}, se recibió {rowptr.size}"
        )
    if (rowptr[:-1] > rowptr[1:]).any():
        raise ValueError("rowptr debe ser no decreciente")

    if colidx.ndim != 1:
        raise ValueError("colidx debe ser un vector 1D")
    if colidx.size == 0:
        return
    if colidx.min() < 0 or colidx.max() >= tile_rows:
        raise ValueError("colidx contiene índices fuera del rango de la tile")


def _format_csr_tile_buffers(
    rowptr: np.ndarray,
    colidx: np.ndarray,
    vals: np.ndarray,
    tile_rows: int,
    nnz_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Normaliza y rellena buffers CSR para ajustarlos al tile del kernel."""

    if nnz_cap <= 0:
        raise ValueError("nnz_cap debe ser mayor que cero")

    rowptr = np.ascontiguousarray(rowptr, dtype=np.int32)
    colidx = np.ascontiguousarray(colidx, dtype=np.int32)
    vals = np.ascontiguousarray(vals, dtype=np.int32)

    _validate_csr_tile(rowptr, colidx, tile_rows)

    nnz = int(rowptr[-1])
    if nnz > nnz_cap:
        raise ValueError(
            f"nnz ({nnz}) excede la capacidad del tile ({nnz_cap})"
        )
    if colidx.size != nnz or vals.size != nnz:
        raise ValueError(
            "El tamaño de colidx/vals debe coincidir con nnz indicado por rowptr"
        )

    colidx_buf = np.zeros(nnz_cap, dtype=np.int32)
    vals_buf = np.zeros(nnz_cap, dtype=np.int32)

    if nnz:
        colidx_buf[:nnz] = colidx
        vals_buf[:nnz] = vals

    return rowptr, colidx_buf, vals_buf, nnz


def make_csr_tile_identity(tile_rows, nnz_cap):
    """Crea una tile identidad y garantiza buffers CSR con padding consistente."""

    if nnz_cap < tile_rows:
        raise ValueError("nnz_cap debe ser al menos igual a tile_rows para la identidad")

    base_rowptr = np.arange(0, tile_rows + 1, dtype=np.int32)  # [0,1,2,...,tile_rows]
    base_colidx = np.arange(0, tile_rows, dtype=np.int32)  # diag
    base_vals = np.ones(tile_rows, dtype=np.int32)

    return _format_csr_tile_buffers(
        base_rowptr,
        base_colidx,
        base_vals,
        tile_rows,
        nnz_cap,
    )

def main(opts):
    # Instrucciones del AIE (según tu flow)
    instr_v = xrt_utils.read_insts(opts.instr)

    # ----- Parámetros fijos del tile (coherentes con el kernel AIE) -----
    TILE_ROWS = 1024
    NNZ_CAP   = 1024 

    BYTES = np.int32().itemsize
    SIZE_ROWPTR = (TILE_ROWS + 1) * BYTES
    SIZE_NNZBUF = NNZ_CAP * BYTES
    SIZE_VEC    = TILE_ROWS * BYTES

    # ----- Dispositivo y kernel -----
    (device, kernel) = test_utils.init_xrt_load_kernel(opts)

    bo_instr  = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1))
    bo_rowptr = xrt.bo(device, SIZE_ROWPTR, xrt.bo.host_only,    kernel.group_id(3))
    bo_colidx = xrt.bo(device, SIZE_NNZBUF, xrt.bo.host_only,    kernel.group_id(4))
    bo_vals   = xrt.bo(device, SIZE_NNZBUF, xrt.bo.host_only,    kernel.group_id(5))
    bo_x      = xrt.bo(device, SIZE_VEC,    xrt.bo.host_only,    kernel.group_id(6))
    bo_y      = xrt.bo(device, SIZE_VEC,    xrt.bo.host_only,    kernel.group_id(7))


    # ----- Datos de prueba: y = x para verificar -----
    rowptr, colidx, vals, nnz_t = make_csr_tile_identity(TILE_ROWS, NNZ_CAP)
    _validate_csr_tile(rowptr, colidx[:nnz_t], TILE_ROWS)
    if opts.verbosity:
        pad_elems = NNZ_CAP - nnz_t
        print(
            f"Tile CSR -> rows: {TILE_ROWS}, nnz: {nnz_t}, padding: {pad_elems}",
            flush=True,
        )
    x_vec = np.arange(1, TILE_ROWS + 1, dtype=np.int32)
    y_out = np.ones(TILE_ROWS, dtype=np.int32)

    # ----- Cargar buffers -----
    bo_instr.write(instr_v, 0)
    bo_rowptr.write(rowptr, 0)
    bo_colidx.write(colidx, 0)
    bo_vals.write(vals, 0)
    bo_x.write(x_vec, 0)
    bo_y.write(y_out, 0)

    # Sync TO_DEVICE
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_rowptr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_colidx.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_vals.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_x.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_y.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    # ----- Ejecutar -----

    opcode = 3  # según tu diseño
    h = kernel(opcode, bo_instr, len(instr_v),
               bo_rowptr, bo_colidx, bo_vals, bo_x, bo_y, bo_trace)
    h.wait()

    # Leer salida
    bo_y.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    bo_trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out = bo_y.read(SIZE_VEC, 0).view(np.int32)[:TILE_ROWS]
    test_utils.write_out_trace(bufTrace, myargs.trace_size, myargs.trace_file)

    # ----- Verificación -----
    errors = 0
    if opts.verify:
        ref = x_vec  # identidad
        errors = int((out != ref).sum())

    if errors == 0:
        print("\nPASS!\n")
        if opts.verbosity:
            print("y[0:8] =", out[:8])
        sys.exit(0)
    else:
        print("\nError count:", errors)
        print("First mismatches (y vs ref):", list(zip(out[:8], ref[:8])))
        print("\nFailed.\n")
        sys.exit(1)

if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
