#export CUDA_VISIBLE_DEVICES=3
import os

if os.path.exists('.local'):
	data_dir = '/home/andrei/work/t7/splited/'
	batch_size = 4
	num_workers = 4
	topk = 2

	SHOW_BAR = False
	DEBUG = True
	TOPk = 3

else:
	#data_dir = '/home/andrei/Data/Datasets/Scales/splited/'
	data_dir = '/mnt/lin2/datasets/splited/'	
	batch_size = 32
	num_workers = 8	

	SHOW_BAR = True
	DEBUG = False
	TOPk = 6
