#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.
This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies
darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with
other scans.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['import_beamline',
           'handle_prep'
           'main']

import argparse
import cohere.utilities.utils as ut
import os
import sys
import importlib
import convertconfig as conv


def handle_prep(experiment_dir, *args, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    # check cofiguration
    print ('preaparing data')
    main_conf_file = os.path.join(experiment_dir, *("conf", "config"))
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        print('Please check the configuration file ' + main_conf_file)
        return None
    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() < main_conf_map['converter_ver']:
        conv.convert(os.path.join(experiment_dir, 'conf'))
        #re-parse config
        main_conf_map = ut.read_config(main_conf_file)
    if 'beamline' in main_conf_map:
        beamline = main_conf_map['beamline']
        try:
            prep = importlib.import_module('beamlines.' + beamline + '.prep')
            det = importlib.import_module('beamlines.' + beamline + '.detectors')
        except Exception as e:
            print(e)
            print('cannot import beamlines.' + beamline + '.prep module.')
            return
    else:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return None
    prep_conf_file = os.path.join(experiment_dir, *("conf", "config_prep"))
    prep_conf_map = ut.read_config(prep_conf_file)
    if prep_conf_map is None:
        print('Please check the configuration file ' + prep_conf_file)
        return None
    if 'data_dir' not in prep_conf_map:
        print('please provide data_dir in configuration file')
        return None
    data_dir = prep_conf_map['data_dir']
    if not os.path.isdir(data_dir):
        print('data directory ' + data_dir + ' is not a valid directory')
        return None

    # create BeamPrepData object defined for the configured beamline
    prep_obj = prep.BeamPrepData(experiment_dir, main_conf_map, prep_conf_map, *args)
    if prep_obj.scan_ranges is None:
        print('no scan given')
        return

    # get directories from prep_obj
    dirs_indexes = prep_obj.get_dirs(data_dir=data_dir)
    if len(dirs_indexes) == 0:
        print('no data found')
        return None

    det_name = prep_obj.get_detector_name()
    if det_name is not None:
        det_obj = det.create_detector(det_name)
        if det_obj is not None:
            prep_obj.set_detector(det_obj, prep_conf_map)
        else:
            print('detector not created')
            return None
    prep_obj.prep_data(dirs_indexes)

    print('done with preprocessing')
    return experiment_dir


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="directory where the configuration files are located")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    handle_prep(experiment_dir)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))