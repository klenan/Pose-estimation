import cv2
import numpy as np
import SimpleITK as itk
from matplotlib import pyplot
import PIL.Image as im
import matplotlib.pyplot as plt
import rvlib as rv
import glob

# Razdalja od tal do kamere z = 1240 mm, debelina elementa = 30 mm

###################################################            Definicije funkcij            ######################################################

def colorToGray(iImage):
    dtype = iImage.dtype
    r = iImage[:,:,0].astype('float')
    g = iImage[:,:,1].astype('float')
    b = iImage[:,:,2].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)

# Nalozi sliko
def loadImage(iPath):
    oImage = np.array(im.open(iPath))
    return oImage

# Prikazi sliko
def showImage(iImage, iTitle=''):
    plt.figure()
    plt.imshow(iImage, cmap = 'gray')
    plt.suptitle(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')

############################################             Kalibracija kamere in priprava slike za nadaljno obddelavo     ###########################################

#*******************VARIABLES**********************
DO_CALIBRATION = 0      # 1 - izvedi kalibracijo; 0 - ne izvedi kalibracije

#naslovi uporabljenih slik
calibImage = 'slike_za_kalibracijo/reference_chessboard.jpg'         # slika šahovnice na katero izvedemo poravnavo

##################################                 Nastavimo sliko, na kateri želimo določiti pozo objekta   test_image1 - test_image5                 #################################

testImage = 'slike_elementov/test_image1.JPG'                        

###########################################################################################################################################################
calibresult = 'calibresult.png'                                      #kam shranš kalibrerano šahovnico
calibObjekt = 'calibObjekt.png'                                      #kam shranš skalibriran objekt
aligned = 'aligned.png'                                              #kam shranš poravnano šahovnico

# Lokacija kamor shranimo kalibrirano sliko
alignedObjekt = 'slike_elementov_kalibrirane/calib_test_image1.png'  #kam shranš poravnan objekt   (za kalibracijo objekta nastav tu (2/2))



#////////////////////////////////////// KALIBRACIJA KAMERE /////////////////////////////////////////
#////////////////////////////////////// ˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇ /////////////////////////////////////////
if DO_CALIBRATION == 1:
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:900:100, 0:700:100].T.reshape(-1,2)  # used to be np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('kameraCalib/*.jpg')
    for fname in images:
            img = cv2.imread(fname)
            #     img = cv2.resize(img, (1296, 972))  # NUJNO, DRGAČ JE PREVELKA RESOLUCIJA SLIKE IN JE NE PRKAŽE
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,7), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners)
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (9,7), corners2, ret)
                    #cv2.imshow('img', img)
                    # cv2.waitKey(10)
            cv2.destroyAllWindows()

    # ***************** kalibracija *****************
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Shranjevanje mtx, dist v datoteko
    np.savetxt('mtx.out', mtx, delimiter = ',')
    np.savetxt('dist.out', dist, delimiter = ',')

# Nalaganje mtx in dist iz datoteke v primeru, da ne izvedemo kalibracije
mtx = np.loadtxt('mtx.out', delimiter = ',')
dist = np.loadtxt('dist.out', delimiter = ',')

# undistortion img
img = cv2.imread(calibImage)
imgObjekt = cv2.imread(testImage)
# img = cv2.resize(img, (1296, 972))
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))    #4. argument: 0 - poreže vse nepotrebne pixle (še kakšnga preveč); 1 - pusti vse potrebne pixle

#kalibracija šahovnice
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(calibresult, dst)
#kalibracija objekta
dstObjekt = cv2.undistort(imgObjekt, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dstObjekt = dstObjekt[y:y+h, x:x+w]
cv2.imwrite(calibObjekt, dstObjekt)

#////////////////////////////////////// PORAVNAVA ŠAHOVNICE NA EKRAN /////////////////////////////////////////
#////////////////////////////////////// ˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇ /////////////////////////////////////////

# ************************izračun ogljišč iz kalibracijske slike ŠAHOVNICE**************************

iCalImage = loadImage(calibresult)
iCalImageG = colorToGray(iCalImage)
resized = cv2.resize(iCalImageG,(648, 486))
# cv2.imshow('iCalImageG', resized)
# cv2.waitKey()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, corners = cv2.findChessboardCorners(iCalImageG, (9,7), None)
corners2 = cv2.cornerSubPix(iCalImageG,corners, (11,11), (-1,-1), criteria)      # v "corners2" so shranjena vsa ogljišča naše šahovnice

points = [(corners2[0,0,0],corners2[0,0,1]),   (corners2[8,0,0],corners2[8,0,1]),   (corners2[62,0,0],corners2[62,0,1]),  (corners2[54,0,0],corners2[54,0,1])]
# print("points: ", points)

# ***************************** KODA *************************************
img = cv2.imread(calibresult)            #šahovnica
imgObjekt = cv2.imread(calibObjekt)      #objekt
rows,cols,ch = img.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                #pretvorimo za plt
imgObjekt = cv2.cvtColor(imgObjekt, cv2.COLOR_BGR2RGB)    #pretvorimo za plt

pts1 = np.float32(points)
# print("pts1:", pts1)
pts2 = np.float32([[100,100], [900,100], [900,700], [100,700]])
# print("pts2:", pts2)

#šahovnica
M = cv2.getPerspectiveTransform(pts1,pts2)   #vzame 2 zbirke točk, vsaka po 4 točke
dst = cv2.warpPerspective(img,M,(1000,800))
#objekt
M = cv2.getPerspectiveTransform(pts1,pts2)   #vzame 2 zbirke točk, vsaka po 4 točke
dstObjekt = cv2.warpPerspective(imgObjekt,M,(1000,800))

plt.subplot(221),plt.imshow(img),plt.title('Input')
plt.subplot(222),plt.imshow(dst),plt.title('Output')
plt.subplot(223),plt.imshow(imgObjekt),plt.title('Input')
plt.subplot(224),plt.imshow(dstObjekt),plt.title('Output')
plt.show()

#šahovnica
dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)        #pretvorimo nazaj za zapis
cv2.imwrite(aligned, dst)
#objekt
dstObjekt = cv2.cvtColor(dstObjekt, cv2.COLOR_RGB2BGR)        #pretvorimo nazaj za zapis
cv2.imwrite(alignedObjekt, dstObjekt)


####################################            Naloži sliko modela in izloči konturo                     #################################################

model_gray = cv2.imread("kontura_noga.png", cv2.IMREAD_GRAYSCALE) # queryiamge
model = np.array(model_gray)
model = cv2.Canny( model, 50, 250 )
im_model, contours_model, hierarchy_model = cv2.findContours(model,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Risanje konture z največjo velikostjo za sliko modela
for i in range(len(contours_model)):
    if contours_model[i].shape[0] > 500:               
        cnt_model = contours_model[i]
        #print('contours_model shape: {}'.format(contours_model[i].shape))  # Poiščemo konture, ki ustrezajo pogoju, ker se model ne spreminja je lahko hardcoded
        
    # Zamenjamo osi arraya in zapišemo v 2D array
arr_model1 = np.swapaxes(contours_model[110],1,2)
arr_model2 = np.swapaxes(contours_model[146],1,2)

arr1 = np.asarray(arr_model1[:,:,0], dtype = 'float64')
arr2 = np.asarray(arr_model2[:,:,0], dtype = 'float64')

    # Združi dva array v enega

krivulja_model = np.concatenate((arr1, arr2), axis = 0)

##########################################                   Zaznaj objekt na sliki             #############################################################

# Naloži kalibrirano sliko
img_color = dstObjekt
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#img = cv2.imread("alignedObjekt9.png", cv2.IMREAD_GRAYSCALE)

# Gamma preslikava za poldarjanje kontrasta
img_gamma = rv.gammaImage(img, 5)
#rv.showImage(img_gamma, 'Gamma')
#pyplot.show()

# Filtriranje slike
img_gauss = cv2.GaussianBlur(img_gamma,(3,3),0, 3)

# Izločanje robov s Canny detektorjem
img_canny = cv2.Canny( img_gauss, 60, 200 )
#rv.showImage(img_canny, 'canny')
#pyplot.show()

# Iz slike robov poišči konture
im_el, contours_el, hierarchy_el = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Izločanje in risanje konture z največjo velikostjo 
for i in range(len(contours_el)):
    if (contours_el[i].shape[0] > 500): 
        cnt_el = contours_el[i]
        cv2.drawContours(img_color, [cnt_el], 0, (0,255,0), 3)
        print('contours_el shape: {}'.format(contours_el[i].shape))

cnt_el = np.swapaxes(cnt_el,1,2)
cnt_el = np.asarray(cnt_el[:,:,0], dtype = 'float64')

#rv.showImage(img, "konture")
#pyplot.show()

###########################################        Poravnaj dve sliki     #################################################

trans_mtx, R_z, center_iptsRef, prestavljen_model = rv.roughAlignment(cnt_el, krivulja_model)

prestavljen_model = np.array(prestavljen_model, dtype = 'uint32')

oMat2D_t, oErr = rv.alignICP(cnt_el[::50], prestavljen_model[::50], iEps=1e-6, iMaxIter=500, plotProgress=False)

############################################      Točke koordinatnega sistema     ######################################

# Točke določimo na podlagi slike konture (modela)
p1 = np.ones((1,3))
p1[0,0] = 677
p1[0,1] = 248

p2 = np.ones((1,3))
p2[0,0] = 583
p2[0,1] = 231

# Translacija
p_ks1 = np.dot(p1, trans_mtx.transpose())
p_ks2 = np.dot(p2, trans_mtx.transpose())

# Rotacija
p_ks1[:,0] = p_ks1[:,0] - center_iptsRef[0,0]
p_ks1[:,1] = p_ks1[:,1] - center_iptsRef[0,1]

p_ks2[:,0] = p_ks2[:,0] - center_iptsRef[0,0]
p_ks2[:,1] = p_ks2[:,1] - center_iptsRef[0,1]

p_ks1 = np.dot(p_ks1, R_z.transpose())
p_ks1[:,0] = p_ks1[:,0] + center_iptsRef[0,0]
p_ks1[:,1] = p_ks1[:,1] + center_iptsRef[0,1]

p_ks2 = np.dot(p_ks2, R_z.transpose())
p_ks2[:,0] = p_ks2[:,0] + center_iptsRef[0,0]
p_ks2[:,1] = p_ks2[:,1] + center_iptsRef[0,1]


# Pomnožimo še z matriko preslikave
p_ks1 = np.dot(p_ks1, oMat2D_t.transpose())
p_ks2 = np.dot(p_ks2, oMat2D_t.transpose())

# Določimo koordinate tretje točke
p_ks3 = np.ones((1,3))

# Rotacija točke p2 za pi/2 okrog p1, da dobimo tretjo koordinato za izris koordinatnega sistema
fi = np.pi/2
R_90 = np.array( [[np.cos(fi), -np.sin(fi), 0], [np.sin(fi), np.cos(fi), 0], [0, 0, 1]] )

p_ks3[:,0] = p_ks2[:,0] - p_ks1[:,0]
p_ks3[:,1] = p_ks2[:,1] - p_ks1[:,1]

p_ks3 = np.dot(p_ks3, R_90.transpose())
p_ks3[:,0] = p_ks3[:,0] + p_ks1[:,0]
p_ks3[:,1] = p_ks3[:,1] + p_ks1[:,1]

#iPtsMov_t = np.dot(rv.addHomCoord2D(prestavljen_model), oMat2D_t.transpose())
#rv.showImage(img_color, 'Slika')
#plt.plot(krivulja_model[::20,0], krivulja_model[::20,1], 'or')    
#plt.plot(prestavljen_model[::20,0], prestavljen_model[::20,1], 'xr')   
#plt.plot(iPtsMov_t[::20,0], iPtsMov_t[::20,1], 'xb') 
#pyplot.show()  

# Zapiši točke v obliki tuple
tmpar1 = np.array(p_ks1)
tmpar1.flatten()
pt1 = (int(round(tmpar1[0][0][0][0])), int(round(tmpar1[0][1][0][0])))
tmpar2 = np.array(p_ks2)
tmpar2.flatten()
pt2 = (int(round(tmpar2[0][0][0][0])), int(round(tmpar2[0][1][0][0])))
pt3 = (int(round(p_ks3[0][0])), int(round(p_ks3[0][1])))

#####################################       Izris končne slike        ############################################

# Izrišemo koordinatni sistem objekta
cv2.line(img_color, pt1, pt2, (255, 0, 0), thickness=3)
cv2.line(img_color, pt1, pt3, (0, 0, 255), thickness=3)

# Izrišemo koordinatni sistem kamere
cv2.line(img_color, (3, 3), (100, 3), (255, 0, 0), thickness=3)
cv2.line(img_color, (3, 3), (3, 100), (0, 0, 255), thickness=3)

# Izračunamo pozicijo koordinatnega sistema objekta
position = pt1

pt1_arr = np.array(pt1)
pt2_arr = np.array(pt2)
vec_obj = pt2_arr - pt1_arr
vec_obj_len = np.sqrt((pt1_arr[0] - pt2_arr[0])**2 + (pt1_arr[1] - pt2_arr[1])**2 )
vec_cam = np.array([100, 0])

# Skalarni produkt
cos_fi = np.dot(vec_obj, vec_cam.transpose()) / (100*vec_obj_len)

# Vektorski produkt za detektiranje ali je kot večji od pi
sin_fi = np.cross(vec_obj, vec_cam) / (100*vec_obj_len)

if sin_fi < 0:
    fi_z = np.arccos(np.abs(cos_fi))
    orientation_z = 180*fi_z/np.pi + 180 
else:
    fi_z = np.arccos(cos_fi)
    orientation_z = 180*fi_z/np.pi

orientation_z = int(round(orientation_z))

# Prikaz končne slike

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_color,'Pozicija lokalnega k.s (mm):',(10,30), font, 0.7,(255,0,0),1,cv2.LINE_AA)
cv2.putText(img_color,"x = {0}".format(position[0]),(350,30), font, 0.7,(255,0,0),2,cv2.LINE_AA)
cv2.putText(img_color,"y = {0}".format(position[1]),(500,30), font, 0.7,(255,0,0),2,cv2.LINE_AA)
cv2.putText(img_color,"z = -1210",(650,30), font, 0.7,(255,0,0),2,cv2.LINE_AA)
cv2.putText(img_color,'Rotacija lokalnega k.s okrog z osi (stopinje):',(10,60), font, 0.7,(0,255,0),1,cv2.LINE_AA)
cv2.putText(img_color,"fi = {0}".format(orientation_z),(550,60), font, 0.7,(0,255,0),2,cv2.LINE_AA)

rv.showImage(img_color, 'Pozicija in orientacija objekta')
pyplot.show()


