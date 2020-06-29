import os, sys, subprocess, shutil, shlex

I = 3
J = 3

# if os.path.isfile('./DATA_MERGER_subproc.py'):
#     print('True')

# sys.exit()


for suffix in ['small', 'large']:
    for i in range(I):
        for j in range(J):
            prefix = str(10*(i+1)) + '_' + str(10*(j+1))

            cmd   = "python3 ./DATA_MERGER_subproc.py" + ' ' + prefix + ' ' + suffix
            args  = shlex.split(cmd)
            p     = subprocess.run(args)


            # allow external program to work
            #p.wait()

            # read the result to a string
            #result_str = p.stdout.read()

            cmd   = "python3 ./NN_CUDA_subproc.py" + ' ' + prefix + ' ' + suffix
            args  = shlex.split(cmd)
            p     = subprocess.run(args)


            # allow external program to work
            #p.wait()

            # read the result to a string
            #result_str = p.stdout.read()






