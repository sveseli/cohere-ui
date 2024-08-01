# Cohere Streaming Demo

## Environment Setup

* Create conda environment and install packages needed for cohere:
```sh
$ sh Miniconda3-latest-Linux-x86_64.sh -b -p CONDA/3.10
$ source CONDA/3.10/etc/profile.d/conda.sh
$ conda create -n cohere
$ conda activate cohere
$ conda install -y python=3.10
$ conda install -y scikit-learn
$ conda install -y tifffile
$ conda install -y -c conda-forge tensorflow-cpu=2.13
$ conda install -y gputil
$ conda install -y psutil
$ conda install -y tqdm
$ conda install -y -c conda-forge mayavi 
$ conda install -y -c conda-forge xrayutilities 
$ conda install -y -c conda-forge pyqt 
$ conda install -y -c conda-forge scikit-image 
$ conda install -y -c conda-forge mpi4py
$ conda install -y -c conda-forge pytorch
```

* Install pvapy package for streaming:
```sh
$ conda install -y -c epics pvapy
```

* Install fabio package for AD sim server support:
```sh
$ pip install fabio
```

* Create conda environment for c2dv (image viewing):

```sh
$ conda create -n c2dv 
$ conda activate c2dv
$ conda install -y python=3.10 # any version of python >= 3.7 should be okay
$ pip install c2dataviewer
$ conda deactivate
```

## Code Checkout

* Checkout cohere and cohere-ui repositories (original cohere repo has
  cohere-ui as submodule):
```sh
$ git clone git@github.com:sveseli/cohere
$ cd cohere/
$ rmdir cohere-ui
$ git clone git@github.com:sveseli/cohere-ui
```

* Prepare config files for example dataset
```sh
$ cd cohere-ui/
$ python setup.py 
```

## Demo

In order to run the demo, one must activate "cohere" environment and
setup PYTHONPATH so that cohere modules can be found.

* Terminal 1, run processing script:
```sh
(cohere) $ cd ..
(cohere) $ export PYTHONPATH=$PWD
(cohere) $ cd cohere-ui/cohere-scripts/
(cohere) $ python process_stream_data.py -in src:image -ws ../example_workspace/scan_54 -bs 201 -out rec:image
Using preprocessor module: <module 'beamlines.aps_34idc.preprocessor' from '/local/sveseli/PVAPY/COHERE2/cohere/cohere-ui/cohere-scripts/beamlines/aps_34idc/preprocessor.py'>
...
```

At this point script will be waiting for data from the source channel
"src:image". Once streaming starts, processed/reconstructed frames will
be available on "rec:image" channel. Note that the script expects all
of the example 201 frames before starting the reconstruction stage.

* Terminal 2, start streaming example dataset:
```sh
$ source CONDA/3.10/etc/profile.d/conda.sh
$ conda activate cohere
(cohere) $ cd cohere/cohere-ui/example_data/AD34idcTIM2_example/Staff20-1a_S0054
(cohere) $ pvapy-ad-sim-server -id . -cs 200 -fps 10 -rp 10 -dc -cn src:image
```

At this point, the processing script will start publihing its output:

```sh
Original frame sum: 4164, beamline pre-processed frame sum: 3685.491452757012
Publishing output frame id 1
Processing frame id 2 (256x256), batch id 0
Original frame sum: 4300, beamline pre-processed frame sum: 3807.5310657022605
Publishing output frame id 2
...
Publishing output frame id 201
Saving beamline pre-processed batch #0 to ../example_workspace/scan_54/preprocessed_data/prep_data_0.tif
Saving standard pre-processed batch #0 to ../example_workspace/scan_54/phasing_data/data_0.tif
data shape torch.Size([256, 256, 240])
torch.float32
------iter 0   error 8.709656331707337e+23
...
```

* Terminal 3, view images on the source and reconstructed channels using
  C2 Data Viewer:
```sh
$ source CONDA/3.10/etc/profile.d/conda.sh 
$ conda activate c2dv
(c2dv) $ c2dv --app image --pv src:image &
(c2dv) $ c2dv --app image --pv rec:image &
```
