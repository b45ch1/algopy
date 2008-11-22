
import pylab as pyl

#SETUP PLOTTING PARAMETERS
fig_width_pt = 350.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.25              # Convert pt to inches
golden_mean = (pyl.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]
legend_padding = 0.05
params = {
	'backend': 'ps',
	'ps.usedistiller': 'xpdf',
	'font.family'  : 'sans-serif',
	'font.style'   : 'normal',
	'font.variant' : 'normal',
	'font.weight'  : 'normal', #bold
	'font.stretch' : 'normal',
	'font.size'    : 'normal', #large
	'axes.labelsize': 11,
	'text.fontsize': 10,
	'title.fontsize':10,
	'legend.fontsize':10,
	'xtick.labelsize': 11,
	'ytick.labelsize': 11,
	'lines.markersize':5,
	'text.usetex': True,
	'figure.figsize': fig_size}
pyl.rcParams.update(params) 
