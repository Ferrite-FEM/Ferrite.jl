using Docile, Lexicon, JuAFEM

const api_directory = "api"

"""
Searches recursively all the modules from packages. As documentation grows, it's a bit
troublesome to add all the new modules manually, so this function searches all the modules
automatically.
Parameters
----------
    module_: Module
        Module where we want to search modules inside
    append_list: Array{Module, 1}
        Array, where we append Modules as we find them
Returns
-------
    None. Void function, which manipulates the append_list
"""
function search_modules!(module_::Module, append_list::Array{Module, 1})
    all_names = names(module_, true)
    for each in all_names
        inner_module = module_.(each)
        if (typeof(inner_module) == Module) && !(inner_module in append_list)
            push!(append_list, inner_module)
            search_modules!(inner_module, append_list)
        end
    end
end

append_list = Array(Module, 0)
search_modules!(JuAFEM, append_list)

const modules = append_list

# main_folder = dirname(dirname(@__FILE__))
# this_folder = dirname(@__FILE__)

# file_ = "README.md"
# run(`cp $main_folder/$file_ $this_folder`)

cd(dirname(@__FILE__)) do
    # Run the doctests *before* we start to generate *any* documentation.
#    for m in modules
#        failures = failed(doctest(m))
#        if !isempty(failures.results)
#            println("\nDoctests failed, aborting commit.\n")
#            display(failures)
#            exit(1) # Bail when doctests fail.
#        end
#    end
    # Generate and save the contents of docstrings as markdown files.
    index  = Index()
    for mod in modules
        Lexicon.update!(index, save(joinpath(api_directory, "$(mod).rst"), mod))
    end
    save(joinpath(api_directory, "index.rst"), index)

    # Add a reminder not to edit the generated files.
  #  open(joinpath(api_directory, "README.md"), "w") do f
  #      print(f, """
  #      Files in this directory are generated using the `build.jl` script. Make
  #      all changes to the originating docstrings/files rather than these ones.
  #      """)
  #  end
    # save(joinpath(api_directory, "index.rst"), index; md_subheader = :category)
    # info("Adding all documentation changes in $(api_directory) to this commit.")
    # success(`git add $(api_directory)`) || exit(1)

end