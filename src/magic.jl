# Creates a static variable, e.g. a variable
# which is initialized at parse time.
# This is good to prevent new allocations in a
# function that need buffer variables
macro static(init)
  var = gensym()
  eval(current_module(), :(const $var = $init))
  var = esc(var)
  quote
    global $var
    $var
  end
end