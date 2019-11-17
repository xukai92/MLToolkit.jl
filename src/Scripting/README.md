## Scripting: helper functions for "single file notebook / script workflow"

The helper functions in this module is to achive Kai's so-called "single file notebook / script workflow", i.e. writing a Jupyter notebook which can be convert to a script by `jupyter nbconvert` and works just fine as in Jupyter.

The functionality is built around the command line argument and dependent excution of statements.

I consider four forms of arguments.

- `argstr`: the argument string that you would use in a command line, e.g. `--x 1 --y 2`
- `argdict`: the dictionary version, e.g. `Dict(:x => 1, :y => 2)`
- `args`: the named tuple version, e.g. `(x=1, y=2)`
- `argstr_flat`: the flatten argument string taht you would use for naming, e.g. `x=1-y=2`

A notebook / script should begin by providing `argstr` (either defined or from the command line; you will still need `ArgParse.jl` for the latter), then parse it to `argdict` which can be modified easily and finally convert to `args` which for actual use in the code. As the code is simply depend on `argstr`, it could be run as a script through the command line interface as well.

There are helper functions to convert an arguments between these forms.

- `argstr` -> `argdict` by `parse_argstr`
- `argdict` -> `argstr` by `argstring`
- `argdict` -> `args` by `process_argdict`
- `argdict` -> `argstr_flat` by `argstring_flat`
  - This is called by `process_argdict` to generate `expname` automatically.

These functions provides some keyword arguments for flexibility, e.g. you can exclude some keys when making a flatten argument string for naming. Check the [source code](https://github.com/xukai92/MLToolkit.jl/blob/master/src/Scripting/args.jl) which should be self-explained. Or the [test file](https://github.com/xukai92/MLToolkit.jl/blob/master/test/Scripting/args.jl) for more concrete examples.

I'd like the same code run in Jupyter and as script, however, there usually some statements you'd like to run only in Jupyter or as a script. Here are two helper functions / macros for this purpose.

- `@jupyter expr`: only executes `expr` if the code is running in Jupyter
- `@script expr`: only executes `expr` if the code is not in Jupyter

There is also the corresponding function (`isjupyter`) available for use.

After converting the notebook into a script, usually you want to run multiple argument combinations. `@sweeprun` is for this and the syntax is as follow.

```julia
sweeprun("sleep @Ts", :T => [1, 2, 3]; maxasync=0)
```

It basically iterates all combinations of the parameters provided (using `maxasync` processes run in the same time; `maxasync=0` means running all combinations in the same time).

> There is a limitation of the current implementation of `sweeprun` which is that it waits for all `maxasync` processes to be finished before running the next batch of `maxasync` processes. This could be improved by using `Distributed`.

#### Other helper functions

You can parse a `.toml` file to `argdict` by `parse_toml`. See [here](https://github.com/xukai92/MLToolkit.jl/blob/master/test/Scripting/Test.toml) for an example file.

You can use `find_latestdir` to find a folder with the lastest datatime format (`ddmmyyyy-H-M-S`). This is useful if you save results as `$argstr_flat/$datatime` for resolve clash of multiple runs of the same experiment. See [here](https://github.com/xukai92/MLToolkit.jl/tree/master/test/Scripting) for an example of folders.

You can use `@tb expr` to execute `expr` only if the current logger is a `TBLogger`. The corresponding function `istb` is also available for use.