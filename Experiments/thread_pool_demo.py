from multiprocessing.pool import ThreadPool
import time

'''
Use apply_async to start the async process
Use get() as a blocking call to get  the results
That's it. 
'''


def foo(bar, baz):
  print 'hello {0}'.format(bar)
  time.sleep(2)
  return 'foo' + baz

pool = ThreadPool(processes=2)

# tuple of args for foo, please note a "," at the end of the arguments
async_result = pool.apply_async(foo, ('world', 'foo',))
async_result2 = pool.apply_async(foo, ('world2', 'foo2',))

# Do some other stuff in the main process

return_val = async_result.get() 

print return_val 
print async_result2.get()

