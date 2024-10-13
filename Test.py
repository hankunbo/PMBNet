import os
import sys

from options.test_options import TestOptions
from network.framework import M1_validation

sys.path.append(".")
sys.path.append("..")

def test():
    opts = TestOptions().parse()
    
    if not os.path.exists(opts.result_frames_path):
        os.mkdir(opts.result_frames_path)
    '''
    if not os.path.exists(opts.result_videos_path):
        os.mkdir(opts.result_videos_path)
    '''
    M1_validation(opts)
    
if __name__ == '__main__':
	test()