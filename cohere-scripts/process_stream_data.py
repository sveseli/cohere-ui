#!/usr/bin/env python

import argparse
import os
import sys
import importlib
import time

import numpy as np
import pvapy as pva
from pvapy.utility.adImageUtility import AdImageUtility

# Cohere imports
import common as cui_common
import cohere_core.utilities as cc_utilities
import cohere_core.data as cc_data
import cohere_core.controller as cc_controller
import cohere_core.controller.phasing as cc_phasing


class CohereDataProcessor:
 
    def __init__(self, input_channel_name, workspace_path, batch_size, output_channel_name):
        self.input_channel = pva.Channel(input_channel_name)
        self.input_channel.monitor(self.process)
        self.workspace_path = workspace_path
        self.config_maps, _ = cui_common.get_config_maps(self.workspace_path, ['config', 'config_prep', 'config_instr', 'config_mp', 'config_data', 'config_rec'])
        self.config_params = {k:v for d in self.config_maps.values() for k,v in d.items()}
        self.preprocessor = self.get_preprocessor()
        self.instrument = self.get_instrument()
        self.bpp_frames = [] # beamline pre-processed frames
        self.batch_size = batch_size
        self.batch_id = 0

        self.pvaServer = None
        self.output_channel_name = output_channel_name
        self.outputFrameId = 0
        if output_channel_name:
            self.pvaServer = pva.PvaServer()
            self.pvaServer.addRecord(output_channel_name, pva.NtNdArray())
            self.pvaServer.start()

    def get_preprocessor(self):
        beamline = self.config_maps['config']['beamline']
        pm = importlib.import_module(f'beamlines.{beamline}.preprocessor')
        print(f'Using preprocessor module: {pm}')
        return pm

    def get_instrument(self):
        beamline = self.config_maps['config']['beamline']
        im = importlib.import_module(f'beamlines.{beamline}.instrument')
        print(f'Using instrument module: {im}')
        i = im.create_instr(self.config_params)
        print(f'Created instrument: {i}')
        return i

    def get_job_size(self, size, method, pc_in_use=False):
        if method is None:
            factor = 170
            const = 100
        elif method == 'ga_fast':
            factor = 184
            const = 428
        elif method == 'populous':
            factor = 250
            const = 0

        # the memory size needed for the operation is in MB
        job_size = size * factor / 1000000. + const
        if pc_in_use:
            job_size = job_size * 2
        return job_size

    def get_ga_method(self):
        rec_config_map = self.config_maps['config_rec']
        ga_method = None
        if rec_config_map.get('ga_generations', 0) > 1:
            ga_method = 'populous'
            if rec_config_map.get('ga_fast'):
                ga_method = 'ga_fast'
        return ga_method

    def get_devices(self, data):
        rec_config_map = self.config_maps['config_rec']
        proc = rec_config_map.get('processing', 'auto')
        devices = rec_config_map.get('device', [-1])
        err_msg, pkg = cui_common.get_pkg(proc, devices)
        if err_msg:
            raise Exception(err_msg)

        if sys.platform == 'darwin' or pkg == 'np':
            devices = [-1]

        want_dev_no = 1
        if devices == [-1]:
            # run locally on cpu
            picked_devs, avail_jobs, hostfile = devices*want_dev_no, want_dev_no, None
        else:
            # based on configured devices find what is available
            # this code below assigns jobs for GPUs
            ga_method = self.get_ga_method()
            job_size = self.get_job_size(data.size, ga_method, 'pc' in rec_config_map['algorithm_sequence'])
            picked_devs, avail_jobs, hostfile = cc_utilities.get_gpu_use(devices, want_dev_no, job_size)

        if hostfile is not None:
            picked_devs = sum(picked_devs, [])
        return pkg, devices

    def process(self, pv):
        (frame_id,raw_frame,nx,ny,nz,color_mode,field_key) = AdImageUtility.reshapeNtNdArray(pv)
        if not nx:
            return
        print(f'Processing frame id {frame_id} ({ny}x{nx}), batch id {self.batch_id}')
        raw_frame = np.swapaxes(raw_frame, 0, 1)
        bpp_frame = self.beamline_preprocess(raw_frame)
        self.update_output_channel(bpp_frame)

        if len(self.bpp_frames) >= self.batch_size:
            # Finalize beamline preprocessing
            bpp_data = self.finalize_beamline_preprocess()
            self.save_beamline_preprocess_file(bpp_data)

            # Run standard preprocessing
            spp_data = self.standard_preprocess(bpp_data)
            self.save_standard_preprocess_file(spp_data)

            # Run reconstruction
            pkg, devices = self.get_devices(spp_data)
            data_rec = cc_phasing.DataRec(self.config_params, spp_data, pkg, self.reconstruction_progress_callback)
            
            self.run_reconstruction(data_rec, devices)
            self.save_reconstruction_results(data_rec)
            rec_data = data_rec.get_rec_data()
            self.update_output_channel(rec_data)
            self.batch_id += 1
            self.bpp_frames = []

    def beamline_preprocess(self, raw_frame):
        bpp_frame = self.instrument.correct_frame(raw_frame)
        self.bpp_frames.append(bpp_frame)
        print(f'Original frame sum: {raw_frame.sum()}, beamline pre-processed frame sum: {bpp_frame.sum()}')
        return bpp_frame

    def finalize_beamline_preprocess(self):
        return np.stack(self.bpp_frames, axis=-1)

    def save_beamline_preprocess_file(self, data):
        save_dir = os.path.join(self.workspace_path, 'preprocessed_data')
        save_file = os.path.join(save_dir, f'prep_data_{self.batch_id}.tif')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f'Saving beamline pre-processed batch #{self.batch_id} to {save_file}')
        cc_utilities.save_tif(data, save_file)

    def standard_preprocess(self, data):
        pp_data = cc_data.prep_data(data, **self.config_params)
        return pp_data
        
    def save_standard_preprocess_file(self, data):
        save_dir = os.path.join(self.workspace_path, 'phasing_data')
        save_file = os.path.join(save_dir, f'data_{self.batch_id}.tif')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f'Saving standard pre-processed batch #{self.batch_id} to {save_file}')
        cc_utilities.save_tif(data, save_file)

    def run_reconstruction(self, rec_controller, devices):
        device = devices[0]
        if rec_controller.init_dev(device) < 0:
            raise Exception (f'Reconstruction failed, device not initialized to {device}')

        if rec_controller.init() < 0:
            raise Exception('Reconstruction failed, check algorithm sequence and triggers in configuration')
    
        if rec_controller.iterate() < 0:
            raise Exception('Reconstruction failed during iterations')

    def save_reconstruction_results(self, rec_controller):
        save_dir = os.path.join(self.workspace_path, f'results_phasing_{self.batch_id}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f'Saving reconstruction results for batch #{self.batch_id} to {save_dir}')
        rec_controller.save_res(save_dir)

    def update_output_channel(self, data):
        if self.output_channel_name:
            if len(data.shape) > 2:
                # Reconstruction, for now publish middle frame
                frameNumber = int(data.shape[0]/2)
                frameData = data[frameNumber]
                print(f'About to publish reconstructed frame number {frameNumber}')
            else:
                # Regular image
                frameData = data
            self.outputFrameId += 1
            print(f'Publishing output frame id {self.outputFrameId}')
            ntndArray = AdImageUtility.generateNtNdArray(self.outputFrameId, frameData)
            self.pvaServer.update(self.output_channel_name, ntndArray)

    def reconstruction_progress_callback(self, rec_data):
        print(f'Current reconstructed sum of all frames: {rec_data.sum()}')
        if self.output_channel_name:
            self.update_output_channel(rec_data)

def main():
    parser = argparse.ArgumentParser(description='Process streamed data')
    parser.add_argument('-in', '--input-channel', required=True, dest='input_channel', help='PVA input channel name')
    parser.add_argument('-out', '--output-channel', dest='output_channel', help='PVA output channel name')
    parser.add_argument('-ws', '--workspace', required=True, dest='workspace', help='Workspace directory')
    parser.add_argument('-bs', '--batch-size', required=True, type=int, dest='batch_size', help='Number of frames per batch')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(f'Unrecognized argument(s): {" ".join(unparsed)}')
        sys.exit(1)
    cdp = CohereDataProcessor(args.input_channel, args.workspace, args.batch_size, args.output_channel)
    try:
        time.sleep(300)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
