#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import popen
from os.path import abspath, dirname, join

sys.path.append(dirname(dirname(abspath(__file__))))
from utils.config_helper import ConfigHelper
from train.helpers.misc_helper import create_hparams

cfg = ConfigHelper()
hparams = create_hparams(cfg)


def main():
    print('Starting server for NMT model...')

    get_pid_list = "ps -ef | sed /grep/d | grep tensorflow_model_server | awk {'print $2'}"
    pid_list = popen(get_pid_list).readlines()
    for pid in pid_list:
        pid = pid[:-1]
        kill_command = 'kill ' + pid
        popen(kill_command)
    print('Former server processes are killed!')

    project_dir = dirname(dirname(abspath(__file__)))
    model_path = join(project_dir, 'exported')
    server_bin_path = join(project_dir, 'bin', 'tensorflow_model_server')
    model_name = hparams.model_name
    port = hparams.port
    server_command = (
        server_bin_path + ' --model_name=' + model_name + ' --port=' + port
        + ' --model_base_path=' + model_path +
        ' --model_version_policy=ALL_VERSIONS' + ' &')
    popen(server_command)


if __name__ == '__main__':
    main()
