def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def showBlankArea(img, blank_row, blank_col, pixel_value, num_cut = 5):
    img[blank_row,:] = pixel_value
    img[:, blank_col] = pixel_value
    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks(np.arange(0, img.shape[1]+1, img.shape[1]/num_cut))
    plt.yticks(np.arange(0, img.shape[0]+1, img.shape[0]/num_cut))
    plt.show()
    return img

def getBlankArea(img, thresholds):
    
    height, width = img.shape
    
    rowSum = np.sum(img, axis = 1) #len(rowSum) = height
    colSum = np.sum(img, axis = 0)
    
    rowSum_nor = rowSum/float(np.max(rowSum))
    colSum_nor = colSum/float(np.max(colSum))
    
    rowArrayVar = np.var(img, axis = 1)
    colArrayVar = np.var(img, axis = 0)
    
    
    blank_row = indices(zip(rowArrayVar,rowSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
    blank_col = indices(zip(colArrayVar,colSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
    
    dimR = len(blank_row)
    dimC = len(blank_col)
    
#     showBlankArea(img, blank_row, blank_col, 122)
    return (dimR * width) + dimC * height - dimR * dimC;

# return segmental blank coverage
def getSegBlankLine(sum_nor, arrayVar, length, thresholds, num_cut):
    
#     step = math.ceil(float(length)/num_cut)
    step = float(length)/num_cut
    result = np.zeros([1, num_cut])
    
#     print "length = ", length
    for i in range(0, num_cut):
        if i*step >= length:
            blank_line = []
        else:
            if (i == (num_cut - 1) or (i+1)*step >= length):
                thisSum_nor = sum_nor[int(round(i*step)) : length]
                thisArrayVar = arrayVar[int(round(i*step)) : length]

            else:
                thisSum_nor = sum_nor[int(round(i*step)) : int(round((i+1)*step))]
                thisArrayVar = arrayVar[int(round(i*step)) : int(round((i+1)*step))]
                
#             print "sum_nor = ", thisSum_nor.shape
            # find the blank row/column by double thresholding
            blank_line = indices(zip(thisArrayVar, thisSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
            
            
        if len(blank_line) == 0:
            result[0, i] = 0
        else:
            result[0, i] = len(blank_line)/float(max(thisSum_nor.shape))
            
    return result


def getBlankLine(img, thresholds, num_cut):
    
    height, width = img.shape
    
    rowSum = np.sum(img, axis = 1) #len(rowSum) = height
    colSum = np.sum(img, axis = 0)
    
    rowSum_nor = rowSum/float(np.max(rowSum))
    colSum_nor = colSum/float(np.max(colSum))
    
    rowArrayVar = np.var(img, axis = 1)
    colArrayVar = np.var(img, axis = 0)
    
        
    result = np.hstack([getSegBlankLine(rowSum_nor, rowArrayVar, height, thresholds, num_cut), getSegBlankLine(colSum_nor, colArrayVar, width, thresholds, num_cut)])

    #Debug mode
#     blank_row = indices(zip(rowArrayVar,rowSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
#     blank_col = indices(zip(colArrayVar,colSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
#     showBlankArea(img, blank_row, blank_col, 122, 5)
#     np.squeeze(np.asarray(b)).tolist()
    return result

def getImageFeatureFromImagePath(imagePath, originalImageInfo, thresholds, num_cut = 5):
     
    data = np.zeros([len(imagePath), 5+num_cut*2])
 
    for (i,filename) in enumerate(imagePath):
         
        id = getImageID(filename)
        img = cv.imread(filename)
         
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
         
        size = img.shape
         
        area_per = float(size[0]*size[1])/originalImageInfo[id]['area']
        height_per = float(size[0])/originalImageInfo[id]['height']
        width_per = float(size[1])/originalImageInfo[id]['width']
        aspect_ratio = float(size[0])/size[1]
        blank_area_per = getBlankArea(img, thresholds)/float(size[0]*size[1])
        segmental_blank_coverage = getBlankLine(img, thresholds, num_cut)
         
        feature_vector = np.hstack([np.asmatrix([area_per, height_per, width_per, aspect_ratio, blank_area_per]), segmental_blank_coverage])
        data[i, :] = feature_vector
         
    print '%d images has been collected.' %len(imagePath)
    return data


def getOriginalImageInfo(originalImageFileList):
    originalImageInfo = {}
    for filename in originalImageFileList:
        info = {}
        img = cv.imread(filename)
        
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        size = img.shape
        info['height'] = size[0]
        info['width'] = size[1]
        info['area'] = size[0]*size[1]
        
        originalImageInfo[getImageID(filename)] = info
        