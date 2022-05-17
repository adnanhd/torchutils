from collections import defaultdict

class ArgumentError(Exception):
    pass

def determine_types(args, kwargs):
    return tuple(type(a) for a in args), \
           tuple((k, type(v)) for k,v in kwargs.items())

function_table = defaultdict(dict)
def overload(*arg_types, **kwarg_types):
    arg_types = tuple(arg_types)
    kwarg_types = tuple(kwarg_types.items())
    def wrap(func):
        func_name = func.__name__
        named_func = function_table[func.__name__]
        named_func[arg_types, kwarg_types] = func
        def call_function_by_signature(*args, **kwargs):
            _args, _kwargs = determine_types(args, kwargs)
            try:
                func = named_func[determine_types(_args, _kwargs)]
            except:
                func_types = [*_args, *tuple(f"{k}:{v}" for k,v in _kwargs)]
                raise ArgumentError('No such overloading whose types to be of the form: ',  
                        f'{func_name}({", ".join(map(str, func_types))})')
            else:
                return func(*args, **kwargs)
        return call_function_by_signature
    return wrap

