import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

fname = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/PMC1197285_pbio.0030314.g004.jpg"
fname = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/PMC193605_pbio.0000019.g005.jpg"
img = cv.imread(fname, 0)

            
if len(img.shape) == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
 
img = np.hstack((np.zeros((img.shape[0], 5), np.float32) +255,img[:, 0:430]))
a = np.ones((5,5),np.float32)
b = np.zeros((5,5),np.float32)
c = np.zeros((15,5),np.float32)
d = np.ones((15,5),np.float32)
f = np.ones((5,15),np.float32)
 
 
def getCorners(img, kernel, threshold, debug = False):
    img_result = cv.filter2D(img, -1, kernel)
    img_result[img_result > threshold] = 255
    dst = cv.cornerHarris(img_result,2,3,0.04)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(img_result,np.float32(centroids),(5,5),(-1,-1),criteria)
    if debug:
        plt.imshow(img_result, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
    return corners

def showCorners(img, corner_list):
    symbols = ["ro", "bo", "yo"]
    for i, corners in enumerate(corner_list):
        plt.plot(corners[:,0], corners[:,1], symbols[i])
    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.show()
    

kernel_split_corner = (1-np.hstack((np.vstack((a,b,a)),c,d)))/225
kernel_top_corner = (1-np.hstack((d, np.vstack((a,b,b)), np.vstack((a,b,a)))))/255
kernel_bottom_corner = (1-np.hstack((d, np.vstack((b,b,a)), np.vstack((a,b,a)))))/255

# kernel_bottom_corner = (1-np.vstack((np.hstack((d, np.vstack((b,b,a)), np.vstack((a,b,a)))),f)))/300

# kernel_top_corner = (1-np.hstack((d,c,np.vstack((b,a,a)))))/225
# kernel_bottom_corner = (1-np.hstack((d,c,np.vstack((a,a,b)))))/225
# kernel_top_corner = (1-np.hstack(d, c, (np.vstack((b, a, a)))))/225
 
# plt.imshow(kernel_bottom_corner, cmap = 'gray', interpolation = 'bicubic')
# plt.show()
 
 
# corner_list = [getCorners(img, kernel_split_corner, 70, debug=True), getCorners(img, kernel_top_corner, 40, debug=True), getCorners(img, kernel_bottom_corner, 50,debug=True)]
# showCorners(img, [getCorners(img, kernel_bottom_corner, 45, debug=True)])


corners = getCorners(img, kernel_bottom_corner, 45)



plt.plot(corners[2,0], corners[2,1], 'ro')
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')


print 'corner'
print (corners[2,0], corners[2,1])
print [int(corners[2,1]), int(corners[2,0])]
print img[int(corners[2,1]), int(corners[2,0])]
print img[int(corners[2,0]), int(corners[2,1])]
end = False
location = [int(corners[2,1]), int(corners[2,0])]

print location
line_length = 5
threshold = 40
isLine = None
print img[location[0]-line_length:location[0],location[1]]

if sum(img[location[0]-line_length:location[0],location[1]]) <= threshold * line_length:
    isLine = True

print isLine

while not end:
    location = location - np.matrix([1,0])
    pixel_value = img[location[0,0], location[0,1]]
    print location, pixel_value
    if pixel_value > 100:
        end = True
    
plt.plot(location[0,1], location[0,0], 'yo')
plt.show()
print location
# 
# plt.plot(split_corner[:,0], split_corner[:,1], 'ro')
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.show()
 
# img_result = cv.filter2D(img, -1, kernel_top_corner)
#  
# # np.vstack((a,b))
# # print img_result<=100
# print np.where(img_result < 50)
# img_result[img_result > 50] = 255
#  
# dst = cv.cornerHarris(img_result,2,3,0.04)
# dst = np.uint8(dst)
# ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv.cornerSubPix(img_result,np.float32(centroids),(5,5),(-1,-1),criteria)
 
 
 
# print "corners"
# print corners
# print img_result[134,216]
# 
# img_result[dst>0.01*dst.max()]=[10]
# # print corners[:,1]
# plt.plot(corners[:,0], corners[:,1], 'ro')
# plt.imshow(img_result, cmap = 'gray', interpolation = 'bicubic')



# print dst
# img2 = cv.drawKeypoints(img, kp, color=(255,0,0))



# sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(img,None)

# img = cv.drawKeypoints(img, kp, img)


# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')


# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.imshow(img_np, interpolation = 'bicubic')
plt.show()