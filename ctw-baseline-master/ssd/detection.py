# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import darknet_tools
import os
import re
import settings
import subprocess

from pythonapi import common_tools


env = {
    'CUDA_VISIBLE_DEVICES': '2',
}


def last_caffemodel(backup_root):
    if not os.path.isdir(backup_root):
        return None
    r = re.compile(r'^.*_iter_(\d+)\.caffemodel$')
    all = [(None, -1)]
    for filename in os.listdir(backup_root):
        filepath = os.path.join(backup_root, filename)
        if os.path.isfile(filepath):
            m = r.match(filename)
            if m:
                i = int(m.group(1))
                all.append((filepath, i))
    return max(all, key=lambda t: t[1])[0]
    
def detect_ssd(split_id, tid):
    exe = os.path.join(settings.CAFFE_ROOT, '.build_release', 'examples', 'ssd', 'ssd_detect.bin')
    deploy = os.path.join(settings.PRODUCTS_ROOT, 'models', 'SSD_512x512', 'deploy.prototxt')
    model = last_caffemodel(os.path.join(settings.PRODUCTS_ROOT, 'models', 'SSD_512x512'))
    test_list = darknet_tools.append_before_ext(settings.TEST_LIST, '.{}'.format(split_id))
    args = [exe, deploy, model, test_list]

    if not os.path.isdir(os.path.dirname(settings.TEST_RESULTS_OUT)):
        os.makedirs(os.path.dirname(settings.TEST_RESULTS_OUT))
    stdout_reopen = darknet_tools.append_before_ext(settings.TEST_RESULTS_OUT, '.{}'.format(split_id))

    new_env = os.environ.copy()
    if 'CUDA_VISIBLE_DEVICES' in new_env:
        env['CUDA_VISIBLE_DEVICES'] = new_env['CUDA_VISIBLE_DEVICES']
    if 1 != settings.TEST_NUM_GPU:
        env['CUDA_VISIBLE_DEVICES'] = '{}'.format(tid % settings.TEST_NUM_GPU+2)
    new_env.update(env)

    for k, v in env.items():
        print('{}={}'.format(k, v), end=' ')
    print(*args)
    p = subprocess.Popen(' '.join(args + ['>{}'.format(stdout_reopen)]), env=new_env, shell=True)
    p.wait()
    assert 0 == p.returncode
