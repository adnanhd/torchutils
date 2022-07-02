#!/usr/bin/env python3
"""! @brief Example Python program with Doxygen style comments."""
##
# @mainpage Doxygen Example Project
#
# @section description_main Description
# PyTorch-Utils is a library to boost the use of PyTorch library using only
# very few lines of codes.
#
# @section changelog_main Changelog
# - Add special project notes here that you want to communicate to the user.
#
# Copyright (c) 2020 Woolsey Workshop.  All rights reserved.
##
# @file doxygen_example.py
#
# @brief Example Python program with Doxygen style comments.
#
# @section description_doxygen_example Description
# Example Python program with Doxygen style comments.
#
# @section libraries_main Libraries/Modules
# - time standard library (https://docs.python.org/3/library/time.html)
#   - Access to sleep function.
# - sensors module (local)
#   - Access to Sensor and TempSensor classes.
#
# @section notes_doxygen_example Notes
# - Comments are Doxygen compatible.
#
# @section todo_doxygen_example TODO
# - None.
#
# @section author_doxygen_example Author(s)
# - Created by John Woolsey on 05/27/2020.
# - Modified by John Woolsey on 06/11/2020.
#
# Copyright (c) 2020 Woolsey Workshop.  All rights reserved.
# Imports

import torchutils.data
import torchutils.models
import torchutils.trainer
import torchutils.logging
import torchutils.metrics

__version__ = torchutils.utils.Version('1.4.2b')
