import Conda
ENVPATH = Conda.ROOTENV
Conda.add("pip")
PIP = joinpath(ENVPATH, "bin/pip")
run(`$PIP install matplotlib tikzplotlib`)

import Pkg
PYTHON = joinpath(ENVPATH, "bin/python")
ENV["PYTHON"] = PYTHON
Pkg.build("PyCall")
