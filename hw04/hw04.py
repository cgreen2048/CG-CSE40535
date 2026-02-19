import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load our pattern
gray = cv2.imread("pattern.png",cv2.IMREAD_GRAYSCALE)

############################################################################
#    Step 1: build the Gabor kernel that will enhance for us vertically oriented patches:
#    cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
#
#    where:
#    ksize  - size of kernel in pixels (n, n), i.e., size of our neighborhood
#    sigma  - size of the Gaussian envelope, i.e., how wide is our Gaussian "hat"
#    theta  - orientation of the normal to the filter's oscilation pattern; e.g., theta = 0.0 means vertical stripes
#    lambda - wavelength of the sinusoidal oscilation; this together with sigma 
#             is resposinble for frequencies enhanced by this filter
#    gamma  - spatial aspect ratio; keep it 1
#    phi    - phase offset; keep it 0
#    ktype  - type and range of values that each pixel in the gabor kernel can hold; keep it cv2.CV_32F

# ***TASK*** Select parameters of your Gabor kernel here:
ksize = 8    # try something between 5 and 15
sigma = 2.8     # try something between 2.0 and 4.0
theta = 0.0     # keep it 0.0 if you want to focus on vertically-oriented patterns 
lbd = 2.0      # try something between 2.0 and 4.0
gamma = 1.0     # keep it 1.0
psi = 0.0       # keep it 0.0

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lbd, gamma, psi, ktype=cv2.CV_32F)

# Normalize the kernel and remove the DC component (do you remember from our class discussion why we are doing this?)
kernel /= kernel.sum()
kernel -= kernel.mean()

# Curious how does the kernel look like? Here we go:
xx, yy = np.mgrid[0:kernel.shape[0], 0:kernel.shape[1]]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, kernel ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
plt.show()


############################################################################
# Step 2: image filtering

res1 = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
cv2.imshow("Filtering result",res1)


############################################################################
# Step 3: image binarization (let's use an idea with maximization of the Fisher ratio, implemeted by Otsu)

th2, res2 = cv2.threshold(res1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Otsu's binarization",res2)


############################################################################
# Step 4: morphological operations and getting your area of interest annotated
# ***TASK*** Choose the type among cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_ERODE or cv2.MORPH_DILATE
# (or a sequence of those, in the order you think makes sense)

se_size = 7  # size of your structuring element (morphological operation kernel) -- try something between 5 and 15
se = np.ones((se_size,se_size), np.uint8)

type = cv2.MORPH_CLOSE
res3 = cv2.morphologyEx(res2, type, kernel=se)

while True:
    cv2.imshow("Areas with vertical pattern annotated",res3)

    action = cv2.waitKey(1)

    if action==27: break 
cv2.destroyAllWindows()