
import cv2
import numpy as np
from PIL import Image
import pytesseract


orgPlaka = cv2.imread("audi.jpg")
#orgPlaka = cv2.imread("focus.png")
#orgPlaka = cv2.imread("honda.jpeg")


    #cv2.imshow("Orijinal foto", orgPlaka)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
grey_Plaka= cv2.cvtColor(orgPlaka, cv2.COLOR_RGB2GRAY)
gurultuazalt = cv2.bilateralFilter(grey_Plaka, 9, 75, 75) #The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. 
cv2.imshow(" gurultuazalt", gurultuazalt) # ilerde canny edge ile kenar bulacağımız için

histogram_e = cv2.equalizeHist(gurultuazalt)    #contrastı artırır
cv2.imshow(" histogram_e", histogram_e)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morfolojikresim = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel, iterations=15)
cv2.namedWindow("5-Morfolojik acilim", cv2.WINDOW_NORMAL)
cv2.imshow("5-Morfolojik acilim", morfolojikresim)

gcikarilmisresim = cv2.subtract(histogram_e, morfolojikresim)  # histogram_e - morfolojikresim
cv2.namedWindow("6-Goruntu cikarma", cv2.WINDOW_NORMAL)
cv2.imshow("6-Goruntu cikarma", gcikarilmisresim)

"""
        cv2.threshold(src, thresh, maxval, type[, dst])
        
    SRC; Giriş dizisidir. Bu dizi gri tonlamalı bir resim olmalıdır.
    TRESH; Eşik ve Piksel değerlerini sınıflandırmak için kullanılır
    MAXVAL; THRESH_BINARY ve THRESH_BINARY_INV eşikleme türlerini maksimum değerde kullanmak için yazılır.
    TYPE; Threshold tipleri belirlenir.
    
    cv2.THRESH_BINARY
    cv2.THRESH_BINARY_INV
    cv2.THRESH_TRUNC
    cv2.THRESH_TOZERO

"""
ret, goruntuesikle = cv2.threshold(gcikarilmisresim, 0, 255, cv2.THRESH_OTSU)
cv2.namedWindow("7-Goruntu Esikleme", cv2.WINDOW_NORMAL)
cv2.imshow("7-Goruntu Esikleme", goruntuesikle)
"""
Sabit bir eşik değeri tüm görüntüler üzerinde kabul edilebilir sonuçlar üretemeyebilir. 
Dolayısıyla eşik değerin, resmin renk dağılımına uygun olarak belirlenmesini sağlayacak bir yönteme ihtiyaç duyulur.
Otsu metodu, gri seviye görüntüler üzerinde uygulanabilen bir eşik tespit yöntemidir.
Bu metod kullanılırken görüntünün arka plan ve ön plan olmak üzere iki renk sınıfından oluştuğu varsayımı yapılır.
Daha sonra tüm eşik değerleri için bu iki renk sınıfının sınıf içi varyans değeri hesaplanır.
Bu değerin en küçük olmasını sağlayan eşik değeri, optimum eşik değeridir.
"""

canny_goruntu = cv2.Canny(goruntuesikle, 250, 255) # kenarları belirler
cv2.namedWindow("8-Canny Edge", cv2.WINDOW_NORMAL)
cv2.imshow("8-Canny Edge", canny_goruntu)

canny_goruntu = cv2.convertScaleAbs(canny_goruntu)
cekirdek = np.ones((3, 3), np.uint8)
gen_goruntu = cv2.dilate(canny_goruntu, cekirdek, iterations=1)
cv2.namedWindow("9-Genisletme", cv2.WINDOW_NORMAL)
cv2.imshow("9-Genisletme", gen_goruntu)
contours, hierarchy = cv2.findContours(gen_goruntu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
# Rakamları alana göre sıralama, böylece sayı plakası ilk 10 konturda olacak
screenCnt = None
for c in contours:
    # yaklaşık çizgi belirliyoruz
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # % 1 bükülme
    # Yaklaşık konturuzun dört noktası varsa, o zaman
    if len(approx) == 4:  # Konturu 4 köşeli olarak seçiyoruz
        print(approx)
        screenCnt = approx
        break
final = cv2.drawContours(orgPlaka, [screenCnt],-1, (0, 0, 255), 2)  # KARENİN RENGİ VE ÇİZİMİ
cv2.namedWindow("10-Konturlu Goruntu", cv2.WINDOW_NORMAL)
cv2.imshow("10-Konturlu Goruntu", final)

mask = np.zeros(grey_Plaka.shape, np.uint8)
yeni_goruntu = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
yeni_goruntu = cv2.bitwise_and(orgPlaka, orgPlaka, mask=mask)
cv2.namedWindow("11-Plaka", cv2.WINDOW_NORMAL)
cv2.imshow("11-Plaka", yeni_goruntu)

kaynak_yolu=""
cv2.imwrite(kaynak_yolu+'plakam.png',yeni_goruntu)

def metin_oku(img_yolu):
    
    img=cv2.imread(img_yolu)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel=np.ones((1,1),np.uint8)
    img=cv2.erode(img,kernel,iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
    cv2.imwrite(kaynak_yolu+'gurultusuz.png',img)
    

    sonuc=pytesseract.image_to_string(Image.open(kaynak_yolu+'gurultusuz.png'),lang='tur')
    return sonuc

print("---------------------------------")
print("metin okuma")
print("---------------------------------")
print(metin_oku('plakam.png'))
text = pytesseract.image_to_string(Image.open("plakam.png"))
print(text)


print("---------------------------------")
print("tamamlandı")

