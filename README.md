# Face Aging

Dieses Repository wurde für das Modul "Aktuelle Themen der künstlichen Intelligenz" des Studiengangs Künstliche Intelligenz an der Technischen Hochschule erarbeitet. Die schriftliche Ausarbeitung inklusive Konferenzvortrag und Code-Präsentation findet ihr auf der [Kurs-Website](https://ki-seminar.github.io/23s/Themen/Face-Aging/).

## GAN

In den Ordner `01_GAN' findet ihr eine Implementierung eines Generative Adversarial Networks mit dem PyTorch-Framework. In den Unterordnern sind in den einzelnen README-Dateien weitere Informationen, z.B. wo ihr die Datensätzen oder Vortrainierte Gewichte downloaden könnt.

## PFA-GAN

In den Ordner `02_PFA-GAN` ist der Code des Papers [PFA-GAN: Progressive Face Aging with Generative Adversarial Network](https://arxiv.org/abs/2012.03459)] implementiert. Orientiert habe ich mich dabei an dieses [Repository](https://github.com/Hzzone/PFA-GAN). Dies ist auch die  offizielle GitHub-Repository zum Paper. In den Unterordnern findet ihr ebenfalls wieder einzelne README-Dateien mit weiteren Informationen zu den Datensätzen, Gewichten und vortraninierten Modellen.

### Nvidia Apex

Zusätzlich wird von Nvidia die [Apex](https://github.com/NVIDIA/apex) Bibliothek benötigt. 

### Dataset

Cross-Age Reference Coding for Age-Invariant Face Recognition and Retrieval

[Reference](https://bcsiriuschen.github.io/CARC/)

### VGG - Age Classifier

VGG implementation from [InterDigitalInc](https://github.com/InterDigitalInc/HRFAE/blob/master/nets.py)

[Model Weights](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html)
