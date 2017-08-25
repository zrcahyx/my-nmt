#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
from os.path import abspath, dirname, exists, join


class ConfigHelper(object):

    def __init__(self, config_file=None):
        self._config_file = config_file

    def get_string_value(self, section, name, config_file='config.conf'):
        if self._config_file:
            config_file = self._config_file
        cf = ConfigParser.ConfigParser()
        cf.read(self._get_cfg_path(config_file))
        value = cf.get(section, name)
        return value

    def get_int_value(self, section, name, config_file='config.conf'):
        if self._config_file:
            config_file = self._config_file
        cf = ConfigParser.ConfigParser()
        cf.read(self._get_cfg_path(config_file))
        value = cf.getint(section, name)
        return value

    def get_float_value(self, section, name, config_file='config.conf'):
        if self._config_file:
            config_file = self._config_file
        cf = ConfigParser.ConfigParser()
        cf.read(self._get_cfg_path(config_file))
        value = cf.getfloat(section, name)
        return value

    def get_boolean_value(self, section, name, config_file='config.conf'):
        if self._config_file:
            config_file = self._config_file
        cf = ConfigParser.ConfigParser()
        cf.read(self._get_cfg_path(config_file))
        value = cf.getboolean(section, name)
        return value

    def set_value(self, section, name, value, config_file='config.conf'):
        if self._config_file:
            config_file = self._config_file
        cf = ConfigParser.ConfigParser()
        path = self._get_cfg_path(config_file)
        cf.read(path)
        cf.set(section, name, value)
        cf.write(open(path, 'w'))
        return value

    def _get_cfg_path(self, config_file):
        config_dir = 'config'
        if not exists(config_dir):
            config_dir = join(dirname(dirname(abspath(__file__))), 'config')
        cfg_path = join(config_dir, config_file)
        return cfg_path
