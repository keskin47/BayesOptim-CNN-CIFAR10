# BayesOptim-CNN-CIFAR10

Bu proje, CIFAR-10 görüntü sınıflandırma problemi için bir **Convolutional Neural Network (CNN)** mimarisinin hiperparametrelerini **Bayesian Optimization (BO)** yaklaşımıyla otomatik olarak optimize eden deneysel bir araştırma çalışmasıdır.

Amaç, CNN’in kapasitesi, düzenlileştirme gücü ve öğrenme dinamiklerini belirleyen hiperparametrelerin yüksek boyutlu arama uzayında **Gaussian Process tabanlı** bir yöntem ile optimize edilmesidir.

---

## Metodoloji

Hiperparametre optimizasyonu **Bayesian Optimization** ile yapılmıştır. Her BO iterasyonunda model:

- 10 epoch hızlı eğitim
- Validation accuracy ölçümü
- Gaussian Process posterior modelleme
- Expected Improvement (EI) ile yeni parametre seçimi

adımlarından geçmiştir.

### Aranan Hiperparametre Uzayı

| Parametre | Aralık |
|----------|--------|
| `filters1` | 16–128 |
| `filters2` | 16–256 |
| `dropout` | 0.0–0.4 |
| `learning_rate` | 1e−5 – 1e−2 |
| `batch_size` | 32–256 |
| `weight_decay` | 1e−6 – 1e−3 |
| `optimizer` | adam / sgd / adamw |

---

## CNN Mimarisi (BO tarafından belirlenen)
Input: 3×32×32

Block 1:
Conv(3 → filters1)
BN → ReLU
Conv(filters1 → filters1)
BN → ReLU
MaxPool(2)

Block 2:
Conv(filters1 → filters2)
BN → ReLU
Conv(filters2 → filters2)
BN → ReLU
MaxPool(2)
Dropout(p=dropout)

Classifier:
Flatten
Linear(filters2·8·8 → 256)
ReLU
Dropout
Linear(256 → 10)


---

## BO Sonuçları

Bayesian Optimization sonucu elde edilen en iyi hiperparametreler:

filters1 = 83
filters2 = 223
dropout = 0.222
learning_rate = 0.00455
batch_size = 251
weight_decay = 0.00043
optimizer = "sgd"


Bu konfigürasyon ile yapılan 30 epoch final eğitimde:

### **Validation Accuracy: 84.23%**

---

## Deneysel Sonuçlar

### **Bayesian Optimization Progress**
![BO Progress](results/bo_progress.png)

### **Training Dynamics (Loss & Accuracy)**
![Training Curves](results/training_curves.png)

### **Hyperparameter Sensitivity (Scatter Plots)**  
Tüm scatter grafikleri:  
`results/bo_hyperparam_scatter/`

Örnek (Learning Rate vs Accuracy):

![lr_vs_acc](results/bo_hyperparam_scatter/learning_rate_vs_acc.png)

---

## Proje Yapısı


---

## Teknik Rapor (PDF)

Tüm matematiksel açıklamalar, BO teorisi, Gaussian Process formülleri, RBF kernel fonksiyonları, mimari analizler ve katman boyut hesaplamaları **teknik rapor** içerisinde sunulmuştur:

-->  **[docs/technical_report.pdf](docs/technical_report.pdf)**

PDF içeriği:

- Gaussian Process tanımı  
- RBF Kernel matematiksel ifadesi  
- BO akış diyagramı  
- Hiperparametre analizleri  
- CNN katmanlarının boyut hesaplamaları  
- Mimarinin blok bazlı açıklaması  

---







