run(`pip install matplotlib`)
PYTHON = replace(read(`which python`, String), "\n" => "")
ENV["PYTHON"] = PYTHON
using Pkg
Pkg.build("PyCall")
