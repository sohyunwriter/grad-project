# Graduation Project (개인 프로젝트)   

RNN 기반 안저이미지 웹어플리케이션 개발   
해당 프로젝트에서 새롭게 접근한 방식은 다음에 있으며, 이 부분에서 이 프로젝트의 의를 찾을 수 있을 것이다.   
1) image classification에 RNN 모델 적용. MNIST dataset에 적용돼 95%의 성능을 보인다는 선행연구를 보고, 안저이미지에도 적용해봤다.   
2) 새로운 데이터셋 및 새로운 모델 적용 및 어플리케이션 개발. FLASK 모듈을 사용해 imagenet dataset까지 적용해본 선행연구는 있지만 아직 안저이미지를 RNN으로 모델링해 웹 어플리케이션을 개발해본 사례는 없다.   
3) 프로젝트를 통한 개인적인 성장      

## 1. System Architecture   
![Untitled Diagram](https://user-images.githubusercontent.com/44013936/80837060-085fbe80-8c31-11ea-8fd7-dc251519957f.png)    

5 multi-class로 구성된 Eye disease dataset을 LSTM 기반 모델로 tranining을 시킨 후, 이 모델을 저장한다. 그리고 이 Keras 모델을 Flask 웹 프레임워크를 이용해 REST API로 배포했다. 이를 통해 USER가 새로운 사진을 upload하면 해당 사진을 predict할 수 있다. 

## 2. Dataset   
KAGGLE DATASET - diabetic-retinopathy-resized    
해당 데이터셋은 총 35,126장이며, 5 multi-class로 구성되어 있다.   
1024*1024로 맞춰있는 resized_train과 resized_train의 noise를 제거한 resized_train_cropped로 구성되어 있는데, 본 연구에서는 resized_train_cropped를 이용했다.   
![히스토그램 안저](https://user-images.githubusercontent.com/44013936/80838107-36de9900-8c33-11ea-957c-c5020688466e.png)   

https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized   

## 3. RNN MODELING   
가장 기저 모델은 다음과 같이 구성했다. 그러나 layer를 더 쌓거나 image size를 더 조절할 수 있다는 점에서 더 향상이 가능해보인다.   
![논문 RNN 모델](https://user-images.githubusercontent.com/44013936/80837053-03027400-8c31-11ea-882c-dc12ded03942.PNG)   

## 4. Web Application
![예측모델](https://user-images.githubusercontent.com/44013936/80837045-fed65680-8c30-11ea-8701-9b73a34d2955.PNG)   

![캡처 RmxRmxmrmx](https://user-images.githubusercontent.com/44013936/80837051-0138b080-8c31-11ea-93ee-b542537ed8d4.PNG)   
