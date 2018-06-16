#ternary_plot.py
import ternary 
import design
import sharma_mittal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_small(d, t, r, ax=None, title=False,
	scale=20, fontsize=9, blw=1.0, msz=6):
	return plot(d, t, r, ax=ax, title=title,
	scale=scale, fontsize=fontsize, blw=blw, msz=msz, small=True)


def plot(d, t, r, ax=None, title=False,
	scale=20, fontsize=20, blw=1.0, msz=8, small=False):
	

	def entropy(p):
		return sharma_mittal.sm_entropy(p, t, r)

	if ax == None: f, ax = plt.subplots()

	tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

	tax.gridlines(color="white", multiple=1, 
		linestyle="-", linewidth=0.1)
	# tax.gridlines(color="white", multiple=2, 
	# 	linestyle="-", linewidth=0.2)

	numTicks = 10
	scaleMtpl = int(scale / numTicks)
	tickLocs = np.append(np.arange(0, scale, scaleMtpl), [scale]) 
	ticks = [round(tick, 2) for tick in np.linspace(0, 1.0, numTicks+1)]
	
	tax.ticks(ticks=ticks, axis='lbr', 
		axes_colors={'l':'grey', 'r':'grey', 'b':'grey'},
		offset=0.02, linewidth=1, 
		locations=list(tickLocs), clockwise=True)
	tax.clear_matplotlib_ticks()
	# ax = plt.gca()
	ax.axis('off')

	tax.left_axis_label(r"$P(k_2) \quad \rightarrow$", 
		fontsize=fontsize, offset=0.12)
	tax.right_axis_label(r"$P(k_1) \quad \rightarrow$",
		fontsize=fontsize, offset=0.12)
	tax.bottom_axis_label(r"$\leftarrow \quad P(k_3)$",
		fontsize=fontsize, offset=0.10)
	tax._redraw_labels()

	cmap = sns.cubehelix_palette(light=1, dark=0, start=2.5, rot=0,
			as_cmap=True, reverse=True)
	tax.heatmapf(entropy, boundary=True, style="triangular", cmap=cmap)
	tax.boundary(linewidth=blw)

	q_color = ['lime', 'fuchsia']
	for i, q in enumerate(d.posterior):
		tax.line(q[0]*scale, q[1]*scale, linewidth=1., 
		marker=['o','o'][i], color='k', linestyle="-", markersize=msz,
		markeredgecolor='k', markerfacecolor=q_color[i], markeredgewidth=1.,
		label=r'$P(K|%s)$' % ['Q_1', 'Q_2'][i])

	tax.line(d.prior*scale, d.prior*scale, linewidth=1., 
		marker='s', color='k', linestyle="-", markersize=msz,
		markeredgecolor='k', markerfacecolor='yellow', markeredgewidth=1.,
		label=r'$P(K)$')

	if not title==False: 
		ax.text(0.5, 1.05, title, 
			ha='center', va='center', 
			transform=ax.transAxes)

	#Create Legend
	if not small:
		ax.text(0.00, 0.95, r'Order $\;\;(r) = %.2f$' % r, 
			ha='left', va='center', transform=ax.transAxes)
		ax.text(0.00, 0.9, r'Degree $(t) = %.2f$' % t, 
			ha='left', va='center', transform=ax.transAxes)

	if small: tax.legend(bbox_to_anchor=(0., 0.00, 1., -.102),
           ncol=3, mode="expand", borderaxespad=0., handletextpad=0,
           fancybox=True, shadow=False, frameon=True)
	else: tax.legend()

	#Remove Colorbar
	f = plt.gcf()
	f.delaxes(f.axes[-1])
	plt.draw()
	return ax


# d = design.design(size=(3,2))
# plot(d,t=1,r=1)
# plt.show()