import os, shutil


for sub_dir in os.listdir('./SOLUTIONS'):

    if sub_dir.endswith('py'):
        continue

    if sub_dir == 'output':
        for sub_sub_dir in os.listdir(os.path.join('./SOLUTIONS', sub_dir)):
            try:
                shutil.rmtree(os.path.join('./SOLUTIONS', sub_dir, sub_sub_dir))
            except NotADirectoryError:
                os.remove(os.path.join('./SOLUTIONS', sub_dir, sub_sub_dir))
    else:
        for fname in os.listdir(os.path.join('./SOLUTIONS', sub_dir)):
            try:
                shutil.rmtree(os.path.join('./SOLUTIONS',sub_dir,fname)) 
            except NotADirectoryError:
                os.remove(os.path.join('./SOLUTIONS',sub_dir,fname))
                
