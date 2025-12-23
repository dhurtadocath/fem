import io
import pstats
import cProfile
from pathlib import Path
from contextlib import redirect_stdout

def make_output_path_obj(func):
    def inner(*args, **kwargs):
         

         
        # getting the returned value
        returned_value = Path(func(*args, **kwargs))

         
        # returning the value to the original frame
        return returned_value
         
    return inner


# def cprofile_function(func):
#     def inner(*args, **kwargs):
         

         
#         # getting the returned value
        

#         pr = cProfile.Profile()
#         pr.enable()
        

        
#         # here goes the code you want to analyze
#         returned_value = func(*args, **kwargs)

#         # 3. Finish the profiling
#         pr.disable()
#         s = io.StringIO()
#         ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
#         with open('profiler_output_for_training_raw_patch_model.txt', 'w') as f:
#             with redirect_stdout(f):
#                 ps.print_stats()
#                 print(s.getvalue())
         
#         # returning the value to the original frame
#         return returned_value
         
#     return inner

def cprofile_function(output_f_name):
    def decorator(function):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            returned_value = function(*args, **kwargs)
                   # 3. Finish the profiling
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            with open(f'{output_f_name}.txt', 'w') as f:
                with redirect_stdout(f):
                    ps.print_stats()
                    print(s.getvalue())
            
            return returned_value
        return wrapper
    return decorator