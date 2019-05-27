# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:47:50 2015

@author: Žiga Špiclin

RVLIB: knjižnica funkcij iz laboratorijskih vaj
       pri predmetu Robotski vid
"""
import numpy as np
import PIL.Image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm # uvozi barvne lestvice
import cv2

def loadImageRaw(iPath, iSize, iFormat):
    '''
    Naloži sliko iz raw datoteke
    
    Parameters
    ----------
    iPath : str 
        Pot do datoteke
    iSize : tuple 
        Velikost slike
    iFormat : str
        Tip vhodnih podatkov
    
    Returns
    ---------
    oImage : numpy array
        Izhodna slika
    
    
    '''
    
    oImage = np.fromfile(iPath, dtype=iFormat) # nalozi raw datoteko
    oImage = np.reshape(oImage, iSize) # uredi v matriko
    
    return oImage


def showImage(iImage, iTitle=''):
    '''
    Prikaže sliko iImage in jo naslovi z iTitle
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    
    Returns
    ---------
    Nothing
    
    
    '''
    plt.figure() # odpri novo prikazno okno
    
    if iImage.ndim == 3 and iImage.shape[0] == 3:
        iImage = np.transpose(iImage,[1,2,0])
    #cv2.imshow(iTitle, iImage)

    plt.imshow(iImage, cmap = cm.Greys_r) # prikazi sliko v novem oknu
    plt.suptitle(iTitle) # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')


def saveImageRaw(iImage, iPath, iFormat):
    '''
    Shrani sliko na disk
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika za shranjevanje
    iPath : str
        Pot in ime datoteke, v katero želimo sliko shraniti
    iFormat : str
        Tip podatkov v matriki slike
    
    Returns
    ---------
    Nothing
    '''
    iImage = iImage.astype(iFormat)
    iImage.tofile(iPath) # zapisi v datoteko


def loadImage(iPath):
    '''
    Naloži sliko v standardnih formatih (bmp, jpg, png, tif, gif, idr.)
    in jo vrni kot matriko
    
    Parameters
    ----------
    iPath - str
        Pot do slike skupaj z imenom
        
    Returns
    ----------
    oImage - numpy.ndarray
        Vrnjena matrična predstavitev slike
    '''
    oImage = np.array(im.open(iPath))
    if oImage.ndim == 3:
        oImage = np.transpose(oImage,[2,0,1])
    elif oImage.ndim == 2:
        oImage = np.transpose(oImage,[1,0])   
    return oImage


def saveImage(iPath, iImage, iFormat):
    '''
    Shrani sliko v standardnem formatu (bmp, jpg, png, tif, gif, idr.)
    
    Parameters
    ----------
    iPath : str
        Pot do slike z željenim imenom slike
    iImage : numpy.ndarray
        Matrična predstavitev slike
    iFormat : str
        Željena končnica za sliko (npr. 'bmp')
    
    Returns
    ---------
    Nothing

    '''
    if iImage.ndim == 3:
        iImage = np.transpose(iImage,[1,2,0])
    elif iImage.ndim ==2:
        iImage = np.transpose(iImage,[1,0])     
    img = im.fromarray(iImage) # ustvari slikovni objekt iz matrike
    img.save(iPath.split('.')[0] + '.' + iFormat)


def drawLine(iImage, iValue, x1, y1, x2, y2):
    ''' Narisi digitalno daljico v sliko

        Parameters
        ----------
        iImage : numpy.ndarray
            Vhodna slika
        iValue : tuple, int
            Vrednost za vrisavanje (barva daljice).
            Uporabi tuple treh elementov za barvno sliko in int za sivinsko sliko
        x1 : int
            Začetna x koordinata daljice
        y1 : int
            Začetna y koordinata daljice
        x2 : int
            Končna x koordinata daljice
        y2 : int
            Končna y koordinata daljice
    '''    
    
    oImage = iImage    
    
    if iImage.ndim == 3:
        assert type(iValue) == tuple, 'Za barvno sliko bi paramter iValue moral biti tuple treh elementov'
        for rgb in range(3):
            drawLine(iImage[rgb,:,:], iValue[rgb], x1, y1, x2, y2)
    
    elif iImage.ndim == 2:
        assert type(iValue) == int, 'Za sivinsko sliko bi paramter iValue moral biti int'
    
        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        napaka = dx - dy
     
        x = x1
        y = y1
        
        while True:
            oImage[y-1, x-1] = iValue
            if x == x2 and y == y2:
                break
            e2 = 2*napaka
            if e2 > -dy:
                napaka = napaka - dy
                x = x + sx
            if e2 < dx:
                napaka = napaka + dx
                y = y + sy
    
    return oImage
    
    
def colorToGray(iImage):
    '''
    Pretvori barvno sliko v sivinsko.
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna barvna slika
        
    Returns
    -------
    oImage : numpy.ndarray
        Sivinska slika
    '''
    dtype = iImage.dtype
    r = iImage[0,:,:].astype('float')
    g = iImage[1,:,:].astype('float')
    b = iImage[2,:,:].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)
    
    
def computeHistogram(iImage, iNumBins, iRange=[], iDisplay=False, iTitle=''):
    '''
    Izracunaj histogram sivinske slike
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna slika, katere histogram želimo izračunati

    iNumBins : int
        Število predalov histograma
        
    iRange : tuple, list
        Minimalna in maksimalna sivinska vrednost 

    iDisplay : bool
        Vklopi/izklopi prikaz histograma v novem oknu

    iTitle : str
        Naslov prikaznega okna
        
    Returns
    -------
    oHist : numpy.ndarray
        Histogram sivinske slike
    oEdges: numpy.ndarray
        Robovi predalov histograma
    '''    
    iImage = np.asarray(iImage)
    iRange = np.asarray(iRange)
    if iRange.size == 2:
        iMin, iMax = iRange
    else:
        iMin, iMax = np.min(iImage), np.max(iImage)
    oEdges = np.linspace(iMin, iMax+1, iNumBins+1)
    oHist = np.zeros([iNumBins,])
    for i in range(iNumBins):
        idx = np.where((iImage >= oEdges[i]) * (iImage < oEdges[i+1]))
        if idx[0].size > 0:
            oHist[i] = idx[0].size
    if iDisplay:
        plt.figure()
        plt.bar(oEdges[:-1], oHist)
        plt.suptitle(iTitle)

    return oHist, oEdges
    
    
def computeContrast(iImages):
    '''
    Izracunaj kontrast slik
    
    Parameters
    ---------
    iImages : list of numpy.ndarray
        Vhodne slike, na katerih želimo izračunati kontrast
        
    Returns : list
        Seznam kontrastov za vsako vhodno sliko
    '''
    oM = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        fmin = np.percentile(iImages[i].flatten(),5)
        fmax = np.percentile(iImages[i].flatten(),95)
        oM[i] = (fmax - fmin)/(fmax + fmin)
    return oM
    
    
def computeEffDynRange(iImages):
    '''
    Izracunaj efektivno dinamicno obmocje
    
    Parameters
    ----------
    iImages : numpy.ndarray
        Vhodne slike
        
    Returns
    --------
    oEDR : float
        Vrednost efektivnega dinamicnega obmocja
    '''
    L = np.zeros((len(iImages,)))
    sig = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        L[i] = np.mean(iImages[i].flatten())
        sig[i] = np.std(iImages[i].flatten())
    oEDR = np.log2((L.max() - L.min())/sig.mean())
    return oEDR
    

def computeSNR(iImage1, iImage2):
    '''
    Vrne razmerje signal/sum
    
    Paramters
    ---------
    iImage1, iImage2 : np.ndarray
        Sliki področij zanimanja, med katerima računamo SNR
        
    Returns
    ---------
    oSNR : float
        Vrednost razmerja signal/sum
    '''
    mu1 = np.mean(iImage1.flatten())
    mu2 = np.mean(iImage2.flatten())
    
    sig1 = np.std(iImage1.flatten())
    sig2 = np.std(iImage2.flatten())
    
    oSNR = np.abs(mu1 - mu2)/np.sqrt(sig1**2 + sig2**2)
            
    return oSNR

def convertRGB2HSV(iImage):
    
    iImage = iImage / 255.0
    r, g, b = iImage[:,:,0], iImage[:,:,1], iImage[:,:,2]
    
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = np.zeros_like(r)
    
    Cmax = np.maximum(r, np.maximum(g,b))
    Cmin = np.minimum(r, np.minimum(g,b))
    delta = Cmax - Cmin + 1e-7 #da nimamo deljenje z 0
    
    h[Cmax == r] = 60.0 * ((g[Cmax == r] - b[Cmax == r])/ delta[Cmax == r] % 6.0)
    h[Cmax == g] = 60.0 * ((b[Cmax == g] - r[Cmax == g])/ delta[Cmax == g] + 2.0)
    h[Cmax == b] = 60.0 * ((r[Cmax == b] - g[Cmax == b])/ delta[Cmax == b] + 4.0)
    
    s[delta != 0.0] = delta[delta!=0.0] / (Cmax[delta!=0.0] + 1e-7)
    
    v = Cmax
    
    #ustvari izhodno sliko
    oImage = np.zeros_like(iImage)
    oImage[:,:,0] = h
    oImage[:,:,1] = s
    oImage[:,:,2] = v
    
    return oImage


def gammaImage(iImage, iGamma):
    '''
    gama preslikava
    '''
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype = 'float')
     
    #prebere mejne vrednosti in obmocje vrednosti
    if iImageType.kind in ('u', 'i'):
        iMaxValue = np.iinfo(iImageType).max
        iMinValue = np.iinfo(iImageType).min
        iRange = iMaxValue - iMinValue
    else:
        iMaxValue = np.max(iImage).max
        iMinValue = np.min(iImage).min
        iRange = iMaxValue - iMinValue
         
        #izvedi gama preslikavo
         
    iImage = (iImage - iMinValue) / float(iRange)
    oImage = iImage ** iGamma
    oImage = float(iRange) * oImage + iMinValue
     
    #zaokrozevanje vrednosti
    if iImageType.kind in ('u', 'i'):
        oImage[oImage < np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage > np.iinfo(iImageType).max] = np.iinfo(iImageType).max
         
    #Vrni sliko v originalnem formatu
    return np.array(oImage, dtype = iImageType)  

def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    """Funkcija za poljubno 2D afino preslikavo"""
    # BEGIN SOLUTION
    iRot = iRot * np.pi / 180
    oMatScale = np.array( ((iScale[0], 0, 0), (0,iScale[1],0), (0,0,1)))
    oMatTrans = np.array( ((1,0,iTrans[0]), (0,1,iTrans[1]), (0,0,1)))
    oMatRot = np.array(((np.cos(iRot), -np.sin(iRot), 0), (np.sin(iRot), np.cos(iRot),0), (0,0,1)))
    oMatShear = np.array(((1, iShear[0],0),(iShear[1],1,0), (0,0,1)))
    oMat2D = np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    
    return oMat2D


def mapAffineApprox2D(iPtsRef, iPtsMov):
    """Afina aproksimacijska poravnava"""
    # YOUR CODE HERE
    iPtsRef = np.matrix(iPtsRef)
    iPtsMov = np.matrix(iPtsMov)
    # Po potrebi dodaj homogeno koordinato
    iPtsRef = addHomCoord2D(iPtsRef)
    iPtsMov = addHomCoord2D(iPtsMov)
    #afina aproksimacija (s psevdoinverzom)
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    #psevdoinverz
    #oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov)) #krajši način zapisa
    
    #psevdoinvez na dolgo in široko
    oMat2D = iPtsRef * iPtsMov.transpose() * np.linalg.inv( iPtsMov * iPtsMov.transpose() )

    return oMat2D


def addHomCoord2D(iPts):
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts

def findCorrespondingPoints(iPtsRef, iPtsMov):
    """Poisci korespondence kot najblizje tocke"""
    # YOUR CODE HERE
    #inicializiraj polje indeksov
    iPtsMov = np.array(iPtsMov)
    iPtsRef = np.array(iPtsRef)
    
    idxPair = -np.ones((iPtsRef.shape[0], 1), dtype= 'int32') #iščemo ujemajoče koordinate
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i, j] = np.sum((iPtsRef[i,:2] - iPtsMov[j, :2])**2)
    #določi bijektivno preslikavo
    while not np.all(idxDist == np.inf):
        i, j = np.where(idxDist == np.min(idxDist))
        idxPair[i[0]] = j[0]
        idxDist[i[0],:] = np.inf #inf = infinity
        idxDist[:, j[0]] = np.inf
    
    #določi pare točk
    idxValid, idxNotValid = np.where(idxPair >= 0)
    idxValid = np.array(idxValid)
    iPtsRef_t = iPtsRef[idxValid, :]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(), :]
    
    return iPtsRef_t, iPtsMov_t


def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=False):
    """Postopek iterativno najblizje tocke"""
    # YOUR CODE HERE
    #inicializiraj izhodne parametre
    curMat = []; oErr = []; iCurIter = 0
    if plotProgress:
        iPtsMov0 = np.matrix(iPtsMov)
        fig = plt.figure()
        ax = fig.add_subplot(111)
    #začni iterativni postopek
    while True:
        #poišči koreospondenčne pare točk
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        #določi afino aproksimacijsko preslikavo
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        #posodobi premične točke
        iPtsMov = np.dot(addHomCoord2D(iPtsMov), oMat2D.transpose())
        #izračunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2] - iPtsMov_t[:,:2])**2)))
        iCurIter = iCurIter + 1
        #preveri kontrolne parametre
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter > iMaxIter or np.all(dMat < iEps):
            break
    
    #določi kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):
        
        if plotProgress:
            iPtsMov_t = np.dot(addHomCoord2D(iPtsMov0), oMat2D.transpose())
            ax.clear()
            ax.plot(iPtsRef[:,0], iPtsRef[:,1],'ob')
            ax.plot(iPtsMov_t[:,0], iPtsMov_t[:,1], 'om')
            fig.canvas.draw()
            plt.pause(1)
        
        oMat2D = np.dot(curMat[i], oMat2D)
    
    return oMat2D, oErr

def roughAlignment(iptsRef, iptsMov):
    '''
    Funkcija ki vzame kot vhod referenčno konturo (iptsRef) in modelirano konturo (iptsMov)
    in nato poravna modelirano konturo na referenčno.

    Kot izhod dobimo translacijsko matriko, rotacijsko matriko, koordinate težišča
    referenčne krivulje in poravnano krivuljo.  
    '''
    # Pretvori v np.array
    iptsMov = np.array((iptsMov), dtype = 'uint32')

    # Določanje težišča krivulj

    center_iptsRef = np.zeros((1,2))
    center_iptsRef[0,0] = np.mean(iptsRef[:,0], axis=None)
    center_iptsRef[0,1] = np.mean(iptsRef[:,1], axis=None)

    center_iptsMov = np.zeros((1,2))
    center_iptsMov[0,0] = np.mean(iptsMov[:,0], axis=None)
    center_iptsMov[0,1] = np.mean(iptsMov[:,1], axis=None)


    # Izračun translacijske matrike med centroma

    iptsMov_3D = np.ones((iptsMov.shape[0],3))
    iptsMov_3D[:,:2] = iptsMov

    tx = center_iptsRef[0,0] - center_iptsMov[0,0]
    ty = center_iptsRef[0,1] - center_iptsMov[0,1]
    trans_mtx = np.array( [[1, 0, tx], [0, 1, ty], [0, 0, 1]] )

    iptsMov_trans = np.dot(iptsMov_3D, trans_mtx.transpose())
    iptsMov_trans = iptsMov_trans[:,:2]   # Translirane točke

    # Poišči najdaljši vektor od težišča za referenco

    r_ref = np.zeros((iptsRef.shape[0]))
    j = 0
    for i in iptsRef:
        x = center_iptsRef[0, 0] - i[0]
        y = center_iptsRef[0, 1] - i[1]
        r_ref[j] = np.sqrt(x**2 + y**2)
        if j != iptsRef.shape[0] - 1:
            j = j+1

    max_index_ref = np.argmax(r_ref)
    far_point = np.zeros((1,2))
    far_point = iptsRef[max_index_ref,:]

    # Poišči najdaljši vektor od težišča za model

    r_mov = np.zeros((iptsMov_trans.shape[0]))
    j = 0
    for i in iptsMov_trans:
        x = center_iptsRef[0, 0] - i[0]
        y = center_iptsRef[0, 1] - i[1]
        r_mov[j] = np.sqrt(x**2 + y**2)
        if j != iptsRef.shape[0] - 1:
            j = j+1
    
    max_index_mov = np.argmax(r_mov)
    far_point_mov = np.zeros((1,2))
    far_point_mov = iptsMov_trans[max_index_mov,:]

    # Določi kot med vektorjema

    vec_ref = far_point - center_iptsRef
    vec_mov = far_point_mov - center_iptsRef

    cos_fi = np.dot(vec_ref, vec_mov.transpose()) / (r_ref[max_index_ref]*r_mov[max_index_mov])
    fi = np.arccos(cos_fi) 
    if fi > np.pi/2:
        fi = fi - np.pi + 0.18  # 0.18 je empirično določena številka, ko je zaznan vektor na napačni strani elementa

    # Rotacijska matrika okrog z osi
    R_z = np.array( [[np.cos(fi), -np.sin(fi), 0], [np.sin(fi), np.cos(fi), 0], [0, 0, 1]] )

    # Rotiran model
    iptsMov_3D_rot = np.ones((iptsMov.shape[0],3)) 
    iptsMov_3D_rot[:,:2] = iptsMov_trans

    iptsMov_3D_rot[:,0] = iptsMov_3D_rot[:,0] - center_iptsRef[0,0]
    iptsMov_3D_rot[:,1] = iptsMov_3D_rot[:,1] - center_iptsRef[0,1]

    iptsMov_rot = np.dot(iptsMov_3D_rot, R_z.transpose())
    iptsMov_rot[:,0] = iptsMov_rot[:,0] + center_iptsRef[0,0]
    iptsMov_rot[:,1] = iptsMov_rot[:,1] + center_iptsRef[0,1]
    iptsMov_rot = iptsMov_rot[:,:2]   # Translirane in rotirane točke

    return trans_mtx, R_z, center_iptsRef, iptsMov_rot

