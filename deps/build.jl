PIP = replace(read(`which pip`, String), "\n" => "")
println("Install Python dependencies using $PIP")
run(`pip install matplotlib tikzplotlib`)
PYTHON = replace(read(`which python`, String), "\n" => "")
ENV["PYTHON"] = PYTHON
using Pkg
Pkg.build("PyCall")
