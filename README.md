# Concept :- 
- In this project, a novel approach is proposed which ensembles machine learning models based upon their confidence for a particular decision.
- If Model 1 is not sure about the decision then Model 1’s decision will not be considered and then Model 2’s confidence will be checked and if found confidence is more than Model 1’s then the decision given by Model 2 will be the final decision and vice-versa
- After comparing with the existing models, the proposed model gave better performance. 
- In this project, two final models have been created which are trained on two different datasets, one using NSL-KDD dataset and other using Canadian Institute of Cyber Security’s dataset.
- We used **CNN (Convolutional Neural Networks)** and ensembled it with **KNN (K-Nearest Neighbors)** for the NSL-KDD Dataset. For CIC IDS 2017 dataset, we used **MLP (Multi-Layer Perceptron)** and ensembled it with **KNN (K-Nearest Neighbors)**. 
- The accuracy of the final model for NSL-KDD dataset is **99.32%** and that for Canadian Institute of Cyber Security’s dataset is **99.68%**. The time required for prediction of a single network packet is **~60 milliseconds**.
- ![image](https://user-images.githubusercontent.com/35119744/163679440-7291125f-f7ed-463b-a2fe-15f2cf99f791.png)


### For whole in detail information:-
https://ieeexplore.ieee.org/document/9076463

### Repository structure:- 
- NSL-KDD folder = Contains application codes and files which basically develops a website where user has to input network packet's data and the application will output the attacks detected.
- out folder = contains the performance results
- Network Packet capturer (Input) :- 
- ![capturer](https://user-images.githubusercontent.com/35119744/163683307-7040b23c-158d-4cb8-82dd-af49aca6c201.JPG)
- Output :- 
- ![attacks](https://user-images.githubusercontent.com/35119744/163683314-29d5a3f1-64ca-4908-8f58-1fdecc03a72f.JPG)
