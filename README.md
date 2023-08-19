# car-motor-truck-classification
This project focuses on categorizing images from car, truck, and motor datasets downloaded from Kaggle and Roboflow. The goal is to employ deep learning methods using PyTorch for accurate classification. The datasets used in this project have been sourced directly from Kaggle and Roboflow

## Overview

- In this project, we looked into using two special models, ResNet-50 and MobileNet V2, to classify images. Our job was to figure out if the pictures showed buildings, forests, glaciers, mountains, seas, or streets.

- For the first model, ResNet-50, it's like a really good starting point. It already knows a lot about images, so we just had to teach it a bit more about our specific pictures.

- The second model, MobileNet V2, is like a super efficient version. It's good at understanding images even if we don't have a lot of computing power.

- We didn't stop there. We also tried something cool called ensemble learning. This means we let both ResNet-50 and MobileNet V2 work together to make decisions. It's like having two experts giving their opinions, and we listen to both because they see things differently.

So, we played with these models, made them smarter for our job, and even let them work together for better results.

## Dataset



## Training Process

#### Transfer learning
In the initial phase of our training process, we harnessed the power of transfer learning. This technique involves leveraging knowledge from pre-trained models and adapting it to our specific task. By utilizing the foundation of existing models that are well-versed in image recognition, we significantly expedited the training process and enhanced the accuracy of our own models. Transfer learning essentially enables us to build upon the expertise accumulated by these models, refined through extensive exposure to diverse images.

In essence, transfer learning empowers us to tap into the domain knowledge acquired by existing models, effectively "transferring" their expertise to our project. This not only accelerates our training process but also heightens the precision of our models in classifying scenes. Their accuracy is boosted by the distinct features they've assimilated from a wide range of images across the domain.

We focused on two models, ResNet-50 and MobileNet V2:

- ResNet-50: ResNet-50 is a deep convolutional neural network architecture known for its exceptional ability to capture intricate features in images. It employs skip connections, allowing for the efficient training of deep networks without succumbing to vanishing gradient problems. We adapted this architecture to our task, enhancing its pre-learned knowledge to excel in our scene classification.


- MobileNet V2: MobileNet V2 stands out for its lightweight design and efficiency, making it suitable for resource-constrained environments. Its depthwise separable convolutions significantly reduce computational demands while preserving accuracy. By repurposing MobileNet V2, we tapped into its streamlined structure for accurate image classification without compromising performance.





#### Ensemble learning

In the second part of our training process, we ventured into ensemble learning. This innovative technique involves combining multiple models to generate predictions, often leading to improved overall performance. The synergy of different models enhances their collective intelligence, enabling them to handle varying decision boundaries and nuances in the data.

Through ensemble learning, we unified the strengths of both ResNet-50 and MobileNet V2, creating a dynamic partnership that increased our models' predictive capabilities. This collaborative approach allows us to capture diverse viewpoints and enhance our ability to accurately classify scenes within the dataset.



#### Different learning rates and batch sizes

Initially, we delved into the impact of different learning rates and batch sizes on our four models. Our analysis revealed intriguing results:

When the learning rate was set at 0.01 and the batch size at 8, the loss reached 0.4. This indicated a favorable combination for minimizing loss.

Conversely, with a learning rate of 0.01 and a batch size of 4, we encountered the highest loss of 0.55. This combination didn't fare as well in terms of minimizing loss.

Shifting our focus to a learning rate of 0.001 paired with a batch size of 8, we observed a notable improvement in loss reduction, falling within the range of 0.3 to 0.35. This demonstrated a promising setup for achieving the lowest loss values.

Similar to the previous case, when we utilized a learning rate of 0.001 and a batch size of 4, the loss further improved, ranging between 0.45 and 0.5.

These findings underscore the importance of choosing the right combination of learning rate and batch size to achieve optimal results in minimizing loss across our models.
