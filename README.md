#  Retail-Analysis

Pipeline Deep Learning end-to-end pour analyser le comportement des passants devant une vitrine à partir de flux vidéo réels.



> 📊 **[Voir la présentation complète du projet]((https://www.canva.com/design/DAG0lTYqmOw/z-EmzJlPZYNW9WjM7s_diw/edit?utm_content=DAG0lTYqmOw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton))**

##  Fonctionnalités

- **Détection et tracking** multi-personnes (YOLO + DeepSORT)
- **Analyse comportementale** : arrêt, ralentissement, passage normal
- **Détection d'entrée/sortie** avec franchissement de seuil intelligent
- **Estimation de pose** et orientation corporelle (body angle)
- **Head pose estimation** (yaw, pitch, roll) via MediaPipe
- **Analyse du regard** vers la vitrine
- **Calcul de métriques métier** : taux d'arrêt, taux d'entrée, score d'attraction
- **Dashboard interactif** ReactJS avec visualisations

##  Architecture

**Backend** : Django REST API  
**Frontend** : Vite + ReactJS + Chart.js  
**Deep Learning** : YOLOv8 (détection, segmentation, pose), RAFT (optical flow), MediaPipe (face mesh)  
**Tracking** : DeepSORT  
**BDD** : MongoDB

##  Métriques Calculées

- **Taux d'arrêt** : % de passants qui s'arrêtent
- **Taux d'entrée** : % de personnes arrêtées qui entrent
- **Temps moyen d'impression** : durée moyenne d'observation de la vitrine
- **Score d'attraction** : `taux_arret × temps_moyen_impression`

##  Stack Technique

- Python 3.8+, PyTorch, Ultralytics, OpenCV, Shapely
- Django, Django REST Framework
- Vite, ReactJS, Chart.js, React Icons
- MongoDB

##  Modèles Deep Learning

**Modèles fine-tunés** :
- **Détection personnes** : YOLOv8 fine-tuné (mAP50: 94.9%, Precision: 92.0%, Recall: 92.5%)
- **Détection visages** : YOLOv8-face Lindev fine-tuné (mAP50: 85.8%, Recall: 84.0%)
- **Segmentation vitrines/portes** : YOLOv8-seg fine-tuné (mAP50(M): 75.4%, Precision: 84.5%)
- **Pose estimation** : YOLOv8-pose fine-tuné (mAP50(B): 69.6%, mAP50(P): 41.9%)

**Modèles pré-entraînés** :
- **Optical Flow** : RAFT-Large (weights=DEFAULT)
- **Face landmarks** : MediaPipe Face Mesh (478 points)

Tous les modèles sont exportés en **ONNX** pour optimiser l'inférence.

## 📄 Licence

MIT

---

**Auteur** : Ons Rekik  
**Stage** : Visshop AI (Juin-Août 2025)
