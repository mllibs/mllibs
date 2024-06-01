import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

args = {}
args['x'] = "radius_worst"
args['y'] = "radius_mean"
args['data'] = nlpi.data['breast_data']['data']
args['hue'] = 'diagnosis'
args['col'] = 'diagnosis'
args['col_wrap'] = None
args['row'] = None
args['mew'] = 1
args['mec'] = 'black'
args['alpha'] = 1
args['s'] = 20
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

'''

Plot scatter plot

	[x] dataframe column string
	[y] dataframe column string
	[data] pd.dataframe of input data
	[hue] dataframe column string
	[mew] marker edge width paremeter int
	[mec] marker edge colour str
	[alpha] opacity of plot float
	[s] size of marker int

'''


def plot_scatterplot(args:dict):

	sns.relplot(x = args['x'], 
		y = args['y'],
		col = args['col'],
		row = args['row'],
		hue = args['hue'], 
		col_wrap = args['col_wrap'],
		alpha = args['alpha'],
		s = args['s'],
		linewidth = args['mew'],
		edgecolor = args['mec'],
		data = args['data'])

	plot_scatterplot(args)


	sns.histplot(
		x=args['x'], 
		y=args['y'],
		hue=args['hue'],
		alpha = args['alpha'],
		linewidth=args['mew'],
		edgecolor=args['mec'],
		data=args['data'],
		bins=50,
		multiple="layer", # dodge, stack, fill
		# element="step",
		fill=False,
		# shrink=.7 # with dodge only
	)

	args = {}
	args['x'] = "island"
	args['y'] = "flipper_length_mm"
	args['data'] = nlpi.data['penguins']['data']
	args['hue'] = None
	args['mew'] = 1
	args['mec'] = 'black'
	args['alpha'] = 1.0
	args['s'] = 20
	pallete = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

	sns.barplot(
		x=args['x'], 
		y=args['y'],
		hue=args['hue'],
		alpha = args['alpha'],
		linewidth=args['mew'],
		edgecolor=args['mec'],
		data=args['data'],
		# col="sex"
		# element="step",
		# fill=False
	)

	px.bar(args['data'],x=args['x'],y=args['y'],color=args['hue'])
	px.histogram(args['data'],x=args['x'],y=args['y'],color=args['hue'],barmode='overlay',nbins=100,template='plotly_white')