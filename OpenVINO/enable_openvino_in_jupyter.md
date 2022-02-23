# how to run openvino inside jupyter notebook?

It provides a quick experiment to get us familiar with OpenVINO API, but make it works took following efforts because OpenVINO's distribution is not so friendly to jupyter notebook:

following solution is only for OpenVINO developers with source code cloned and required customized rebuild.
## 1 add python folder to `import path` for import to find it

https://realpython.com/python-import/#pythons-import-path
 
   - modifying PYTHONPATH in setupvars.sh OR
   - appending new path to sys.path

but none works for `Pylance` vs-code-extension to recognize, so I copied setup.py from source tree to install/python/python3.6 folder and made a few modification(remove cmake to generate .so files), then use `pip3 install -e .`, it works. 

```python
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import Extension, setup
__version__ = os.environ.get("NGRAPH_VERSION", "0.0.0.dev0")

packages = [
    "ngraph",
    "ngraph.opset1",
    "ngraph.opset2",
    "ngraph.opset3",
    "ngraph.opset4",
    "ngraph.opset5",
    "ngraph.opset6",
    "ngraph.opset7",
    "ngraph.opset8",
    "ngraph.utils",
    "ngraph.impl",
    "ngraph.impl.op",
    "ngraph.impl.op.util",
    "ngraph.impl.passes",
    "openvino",
    "openvino.frontend",
    "openvino.offline_transformations",
    "openvino.runtime.opset1",
    "openvino.runtime.opset2",
    "openvino.runtime.opset3",
    "openvino.runtime.opset4",
    "openvino.runtime.opset5",
    "openvino.runtime.opset6",
    "openvino.runtime.opset7",
    "openvino.runtime.opset8",
    "openvino.runtime.utils",
    "openvino.runtime.op",
    "openvino.runtime.op.util",
    "openvino.runtime.passes",
]

data_files = []

with open("requirements.txt") as req:
    requirements = req.read().splitlines()

setup(
    name="openvino",
    description="OpenVINO - deploying pre-trained deep learning models",
    version=__version__,
    author="Intel Corporation",
    url="https://github.com/openvinotoolkit/openvino",
    license="License :: OSI Approved :: Apache Software License",
    packages=packages,
    install_requires=requirements,
    data_files=data_files,
    zip_safe=False,
    extras_require={},
)
```

## 2 allow dynamic linker to find shared objects files:

http://blog.tremily.us/posts/rpath/ gives a detailed explanation, to summarize:

 - LD_LIBRARY_PATH (ugly, sometime nowhere to set)
 - ldconfig (system-wide permenent) https://linux.101hacks.com/unix/ldconfig
 - RPATH (recommended, relative path is better)
 - RUNPATH (better, allows overriding by LD_LIBRARY_PATH)

We saw other python packages are exactly using this methods: 
> `readelf -d cv2.cpython-36m-x86_64-linux-gnu.so` shows relative RPATH like [$ORIGIN/../opencv_python.libs].

But so files installed by OpenVINO `make install` has no RPATH at all, we saw following build log:

```bash
-- Installing: /home/hddl/openvino/build/install/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so
-- Set runtime path of "/home/hddl/openvino/build/install/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so" to ""
```

This is default cmake behaviour according to https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#default-rpath-settings

This can be fixed by adding CMAKE_INSTALL_RPATH:
```bash
cmake -DCMAKE_INSTALL_RPATH=/home/hddl/openvino/build/install/runtime/3rdparty/tbb/lib:/home/hddl/openvino/build/install/runtime/3rdparty/hddl/lib:/home/hddl/openvino/build/install/runtime/lib/intel64 .
```

RPATH support in CMake is described in:
 - https://dev.my-gate.net/2021/08/04/understanding-rpath-with-cmake/
 - https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
