import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
import re


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

#  Bu kod Fotoğraf dosyalarının bulunduğu yer
image_folder = 'dataset'  

# Dizin içindeki tüm fotoğraflarda gez
for filename in os.listdir(image_folder):
    # Dosya uzantısını kontrol et (sadece .jpg, .png gibi görselleri işle)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(image_folder, filename)

        # Görüntüyü yükle
        img = cv2.imread(img_path)
        if img is None:
            print(f"{filename} dosyası yüklenemedi.")
            continue

        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        # Görüntü iyileştirme
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Kenar tespiti
        edges = cv2.Canny(gray, 100, 200)

       
        # Plaka varsa onun tespitini yaparız
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

        # Plakaları tespit et
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 10))

        # Plakaları görüntüde işaretle
        for (x, y, w, h) in plates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Plaka", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Plaka bölgesinden metni çıkart
        plaka_text = ""
        for (x, y, w, h) in plates:
            plaka_region = img[y:y + h, x:x + w]

            # Plaka bölgesinden metni çıkart
            custom_config = r'--oem 3 --psm 6'  #Tesseract OCR motorunun kullanacağı motoru belirtir.
            plaka_text = pytesseract.image_to_string(plaka_region, config=custom_config)
            plaka_text = plaka_text.strip()
            plaka_text = re.sub(r'[^A-Za-z0-9]', '', plaka_text)

            print(f"{filename} - Kesilmiş Plaka Metni:", plaka_text)

        # 3 görseli aynı grafikte göster
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Kenar tespiti görseli axes burda grafik alanını tespit ediyor
        axes[0].imshow(edges, cmap='gray')
        axes[0].set_title(f"{filename} - Kenar Tespiti")
        axes[0].axis('off')

        # Plaka tespitli fotoğraf
        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"{filename} - Plaka Tespiti")
        axes[1].axis('off')

        # Plaka bölgesini göster
        if len(plates) > 0:  # Eğer plaka tespit edilmişse
            for (x, y, w, h) in plates:
                plaka_region = img[y:y + h, x:x + w]
                axes[2].imshow(cv2.cvtColor(plaka_region, cv2.COLOR_BGR2RGB))
                axes[2].set_title(f"{filename} - Plaka Bölgesi")
                axes[2].axis('off')
        else:
            axes[2].axis('off')  # Eğer plaka yoksa, eksik görseli gizle

        plt.tight_layout()
        plt.show()
