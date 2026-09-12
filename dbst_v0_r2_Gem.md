# Decay-Seamed Block Thomas v0 - review 2

##

Ah, the joys of working on the bleeding edge of Python GPU compilers!

You hit a known limitation: unlike C++ CUDA, Numba does not natively expose the `%clock64` hardware register to the Python API.

However, because this is a **Phase 0 throwaway diagnostic script**, we can use the "Numba Dark Arts"—injecting raw PTX assembly directly into the LLVM IR compiler.

Here is the exact fix. You just need to add a custom `@intrinsic` compiler extension at the top of your script.

### The Fix for `dbg_serial_split_v0.py`

**1. Add these imports at the top of your file (right under `from numba import cuda`):**

```python
from numba.core import types
from numba.cuda.extending import intrinsic
from llvmlite import ir
```

**2. Define the PTX assembly injector just below your imports:**

```python
@intrinsic
def clock64(typingctx):
    """Injects raw PTX assembly to read the GPU's 64-bit cycle counter."""
    sig = types.int64()
    def codegen(context, builder, sig, args):
        # Define LLVM function type: returns 64-bit int, takes no args
        fty = ir.FunctionType(ir.IntType(64), [])
        # Inline PTX: move special register %clock64 into output constraint '=l' (64-bit int)
        asm = ir.InlineAsm(fty, "mov.u64 $0, %clock64;", "=l", side_effect=True)
        return builder.call(asm, [])
    return sig, codegen
```

**3. Search and Replace in your kernel:**
Replace all instances of `cuda.clock64()` with `clock64()` inside `_kern`. 
*(e.g., `t0 = cuda.clock64()` becomes `t0 = clock64()`)*.

---

### Why this works

Numba compiles Python to LLVM IR, which the NVVM compiler then lowers to PTX for the GPU. The `@intrinsic` decorator allows us to bypass Python entirely and hand LLVM a raw string of PTX assembly (`mov.u64 $0, %clock64;`). 

Because we flag it with `side_effect=True`, the LLVM optimizer is forced to leave it exactly where we placed it (preventing the compiler from re-ordering our timestamps or optimizing them away).

Run it again with those changes. It will compile cleanly and you will get a beautiful, cycle-accurate printout of exactly where the GPU is spending its time inside that block!
