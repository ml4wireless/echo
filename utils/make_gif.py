import numpy as np
import os
import imageio
import sys

def save_gif(job_name='sample_job', echo=False):
	# for echo echo
	if not echo:
		png_dir = '%s/plots'%job_name#/plot_%s--0'%(job_name,job_name)
	# for echo
	else:
		png_dir = '../events_clean_replicate_colin/plots_for_presentation_8-1-18'


	files = ["%s/%s"%(png_dir,f) for f in os.listdir(png_dir)]
	files.sort(key = lambda x: os.path.getmtime(x))
	a1d, a1m, a2d, a2m, cal1, cal2 = [], [], [], [], [], []
	for which, l in zip(['A1D', 'A1M', 'A2D', 'A2M', 'cal1', 'cal2'],
						[a1d, a1m, a2d, a2m, cal1, cal2]):
		images = []
		for file_name in files:
		    if file_name.endswith('.png') and which in file_name:
		        l.append(imageio.imread(file_name))

	le = len(a1d)
	for l in [a1d, a1m, a2d, a2m, cal1, cal2]:
		assert len(l) == le, "%s, %s" % (str(len(l)), str(le))
	if not echo:
		np_row = 480
		np_col = 640
	else:
		np_row = 288
		np_col = 432
	np_rows = np_row*2 + 20
	np_cols = np_col*3 + 30

	pics = []
	for i in range(len(a1d)):
		pic = np.zeros((np_rows, np_cols, 4)).astype(int)+255
		pic[5:np_row+5, 5:np_col+5] = a1m[i][:,:]
		pic[15+np_row:2*np_row+15, 5:np_col+5] = a2m[i][:,:]
		if echo:
			pic[5:np_row+5, 15+np_col:2*np_col+15] = a2d[i][:,:]
			pic[15+np_row:2*np_row+15, 15+np_col:2*np_col+15] = a1d[i][:,:]
		else:
			pic[5:np_row+5, 15+np_col:2*np_col+15] = a1d[i][:,:]
			pic[15+np_row:2*np_row+15, 15+np_col:2*np_col+15] = a2d[i][:,:]
		pic[5:np_row+5, 25+np_col*2:3*np_col+25] = cal1[i][:,:]
		pic[15+np_row:2*np_row+15, 25+np_col*2:3*np_col+25] = cal2[i][:,:]
		pics.append(pic)

	# for echo echo
	if not echo:
		imageio.mimsave('%s/all.gif'%job_name, pics, duration=0.1)
	# for echo
	else:
		imageio.mimsave('../events_clean_replicate_colin/sample_%s.gif'%"all", pics, duration=0.3)

def save_final(job_name='sample_job', echo=False):


	# for echo echo
	if not echo:
		png_dir = '%s/plots'%job_name#/plot_%s--0'%(job_name, job_name)
	# for echo
	else:
		png_dir = '../events_clean_replicate_colin/plots_for_presentation_8-1-18'


	files = ["%s/%s"%(png_dir,f) for f in os.listdir(png_dir)]
	files.sort(key = lambda x: os.path.getmtime(x))
	a1d, a1m, a2d, a2m, cal1, cal2 = [], [], [], [], [], []
	for which, l in zip(['A1D', 'A1M', 'A2D', 'A2M', 'cal1', 'cal2'],
						[a1d, a1m, a2d, a2m, cal1, cal2]):
		images = []
		for file_name in files:
		    if file_name.endswith('.png') and which in file_name and "with" not in file_name:
		        l.append(imageio.imread(file_name))

	# le = len(a1d)
	# for l in [a1d, a1m, a2d, a2m, cal1, cal2]:
	# 	assert len(l) == le
	if not echo:
		np_row = 480
		np_col = 640
	else:
		np_row = 288
		np_col = 432
	np_rows = np_row*2 + 20
	np_cols = np_col*3 + 30


	pic = np.zeros((np_rows, np_cols, 4)).astype(int)+255
	pic[5:np_row+5, 5:np_col+5] = a1m[-1][:,:]
	pic[15+np_row:2*np_row+15, 5:np_col+5] = a2m[-1][:,:]
	if echo:
	    pic[5:np_row+5, 15+np_col:2*np_col+15] = a2d[-1][:,:]
	    pic[15+np_row:2*np_row+15, 15+np_col:2*np_col+15] = a1d[-1][:,:]
	else:
	    pic[5:np_row+5, 15+np_col:2*np_col+15] = a1d[-1][:,:]
	    pic[15+np_row:2*np_row+15, 15+np_col:2*np_col+15] = a2d[-1][:,:]
	pic[5:np_row+5, 25+np_col*2:3*np_col+25] = cal1[-1][:,:]
	pic[15+np_row:2*np_row+15, 25+np_col*2:3*np_col+25] = cal2[-1][:,:]
	imageio.imwrite('%s/end.png'%job_name, pic)

	# # for echo echo
	# if not echo:
	# 	imageio.mimsave('%s/plots/sample_%s.gif'%(job_name,"all"), pics, duration=0.3)
	# # for echo
	# else:
	# 	imageio.mimsave('../events_clean_replicate_colin/sample_%s.gif'%"all", pics, duration=0.3)