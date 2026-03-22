# Cyber GPT 🤖

Transformer mimarisini sıfırdan PyTorch ile yazdım ve eğittim.

## Ne yaptım?
- Multi-head attention mekanizmasını sıfırdan kodladım
- Google Colab'de kendi verim ile eğittim
- FastAPI ile REST API kurdum
- HTML/JS ile siber temalı chat arayüzü yaptım

## Nasıl Çalışır?
Kullanıcı chat arayüzünden mesaj gönderir. FastAPI bu mesajı alır, transformer modeline verir, model karakter karakter yeni metin üretir ve cevabı geri döner.

## Öğrendiklerim
- Transformer ve attention mekanizması
- PyTorch ile model eğitimi
- REST API tasarımı
- Model dosyası entegrasyonu

## Teknolojiler
- PyTorch
- FastAPI
- Python
- HTML/JS

## Nasıl Çalıştırılır?
pip install fastapi uvicorn torch
python son.py
Tarayıcıda http://127.0.0.1:8000 adresini aç

## Not
Model küçük bir veri setiyle eğitildiği için tutarlı cevaplar üretemiyor. 
Bu proje bir ürün değil, transformer mimarisini sıfırdan öğrenmek için yapılmış 
bir çalışmadır.
