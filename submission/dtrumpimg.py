import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fname = 'thumbsup'
cmap = 'gist_stern'
t1 = mpimg.imread('t'+fname+'.jpg')
fig = plt.figure(frameon=False)
ax = plt.Axes(fig,[0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(t1[:,:,0],cmap=cmap)
fig.savefig('T2'+fname+'.png',bbox_inches='tight',pad_inches=0)
