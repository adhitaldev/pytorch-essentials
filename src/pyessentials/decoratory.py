"""
Functions are first class objects i Python and can be pased as arguments to
other functions, and can be defined inside other functions.
Decorators are often used to check access usage, api key checks, adding delay 
for rate-limiting, etc.
"""
import timeit
import functools

def func_time_decorator(f):
    """
    Calculates the time that a function takes to 
    execute
    """
    def wrapper(*args, **kwargs):
        begin = timeit.default_timer()
        result = f(*args, **kwargs)
        end = timeit.default_timer()
        print(f'Decorator: It took {end - begin}s  for function to execute')
        return result
    return wrapper

"""
@func_time_decorator here generally translates to soft_function = @func_time_decorator(sort_function)
"""
@func_time_decorator
def sort_function_decorator(nums):
    return sorted(nums)

"""
Without decorator usage
"""
def sort_function_traditional(nums):
    return sorted(nums)

"""
Without decorator the same functionality can be achieved by using functools.partial
which takes in a function and positional args to it and creates a new function whose functions are 
"""
def func_time_traditional(f):
    begin = timeit.default_timer()
    result = f()
    end = timeit.default_timer()
    print(f"Traditional: It took {end - begin}s for function to execute")
    return result


def run_example():
    """
    Runs various examples  of the decorator functionality
    """
    #With decorator usage
    print(f"Using decorators")
    sort_function_decorator([i for i in range(100)])
    sort_function_decorator([i for i in range(100000)])
    sort_function_decorator([i for i in range(10000000)])
    #Without decorator usage
    print(f"Without decorators")
    func_time_traditional(functools.partial(sort_function_traditional,[i for i in range(100)]))
    func_time_traditional(functools.partial(sort_function_traditional,[i for i in range(100000)]))
    func_time_traditional(functools.partial(sort_function_traditional,[i for i in range(100000000)]))
    

if __name__ == "__main__":
    run_example()
