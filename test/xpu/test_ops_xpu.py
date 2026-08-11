# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import functools
import sys
import unittest.mock as mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_ops import (
        fake_autocast_device_skips,
        TestCommon,
        TestCompositeCompliance,
        TestFakeTensor,
        TestForwardADWithScalars,
        TestMathBits,
    )

fake_autocast_device_skips["xpu"] = {"linalg.pinv", "pinverse"}
instantiate_device_type_tests(TestCommon, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu", allow_xpu=True)
# in finegrand
instantiate_device_type_tests(
    TestCompositeCompliance, globals(), only_for="xpu", allow_xpu=True
)
# only CPU
# instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="xpu", allow_xpu=True)
# not important
instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestForwardADWithScalars, globals(), only_for="xpu", allow_xpu=True
)
# instantiate_device_type_tests(TestTags, globals(), only_for="xpu", allow_xpu=True)

# On XPU devices with <=64KB local memory (SLM), double-precision oneMKL FFT kernels
# exhaust device SLM and raise UR_RESULT_ERROR_OUT_OF_RESOURCES for complex128.
# Wrap every test_dtypes_* method to silently skip complex128 on such devices.
_xpu_local_mem = 0
if hasattr(torch, "xpu") and torch.xpu.is_available():
    try:
        _xpu_local_mem = getattr(
            torch.xpu.get_device_properties(torch.device("xpu")), "local_mem_size", 0
        )
    except Exception:
        pass

if _xpu_local_mem > 0 and _xpu_local_mem <= 64 * 1024:
    _test_ops_mod = sys.modules.get("test_ops")
    if _test_ops_mod is not None:
        _orig_ataca = _test_ops_mod.all_types_and_complex_and

        def _ataca_no_c128(*args):
            return tuple(t for t in _orig_ataca(*args) if t != torch.complex128)

        TestCommonXPU = globals().get("TestCommonXPU")
        if TestCommonXPU is not None:
            _name = "test_dtypes_stft_xpu"
            if hasattr(TestCommonXPU, _name):
                _orig_method = getattr(TestCommonXPU, _name)

                def _make_wrapper(fn):
                    @functools.wraps(fn)
                    def _wrapper(self):
                        with mock.patch.object(
                            _test_ops_mod,
                            "all_types_and_complex_and",
                            _ataca_no_c128,
                        ):
                            return fn(self)

                    return _wrapper

                setattr(TestCommonXPU, _name, _make_wrapper(_orig_method))


if __name__ == "__main__":
    run_tests()
