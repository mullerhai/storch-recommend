# storch-recommend
   Welcome to the ‌storch-recommend‌ repository, a Scala-based deep learning project dedicated to building recommendation systems. Leveraging the power of deep learning, this project aims to provide a comprehensive suite of layers and models tailored for recommendation tasks.
The storch-recommend project is a Scala-based implementation that offers a comprehensive suite of recommendation models and layers. Leveraging the Storch framework, it enables developers to build diverse recommendation systems tailored to different use - cases. This README provides an in - depth look at all the available models and layers, along with instructions on how to utilize them effectively.

### sbt repo
```
libraryDependencies += "io.github.mullerhai" % "storch-recommend_3" % "0.1.0"
```

### Project URL
https://github.com/mullerhai/storch-recommend

### Overview
The ‌storch-recommend‌ project encompasses ‌16 layers‌ and ‌23 models‌, each meticulously designed to cater to various aspects of recommendation systems. These components collectively enable the development of sophisticated and high-performing recommendation engines.

## Project Structure
The project is mainly divided into two parts: model and layer .

### Model
The model directory contains various recommendation models, including:

- FieldAwareNeuralFactorizationMachineModel : A field-aware neural factorization machine model.
- FieldAwareFactorizationMachineModel : A field-aware factorization machine model.
- AutomaticFeatureInteractionModel : A model for automatic feature interaction.
- ProductNeuralNetworkModel : A product neural network model.
- AttentionalFactorizationMachineModel : An attentional factorization machine model.
- WideAndDeepModel : A wide and deep model.
- FactorizationSupportedNeuralNetworkModel : A factorization-supported neural network model.
- NeuralFactorizationMachineModel : A neural factorization machine model.
- AITMModel
- MMoEModel
- OMoEModel
- PLEModel
- SharedBottomModel
- SingleTaskModel
- DSSM
- DeepFactorizationMachineModel
- DeepCrossNetworkModel
- HighOrderFactorizationMachineModel
- AdaptiveFactorizationNetwork
- LogisticRegressionModel
- NeuralCollaborativeFiltering
### Layer
The layer directory contains various layers used in the models, including:

- FeaturesEmbedding : A layer for feature embedding.
- MultiLayerPerceptron : A multi-layer perceptron layer.
- AttentionalFactorizationMachine : An attentional factorization machine layer.
- CompressedInteractionNetwork : A compressed interaction network layer.
- OuterProductNetwork : An outer product network layer.
- AnovaKernel : An ANOVA kernel layer.
- CrossNetwork
- EmbeddingModule
- FactorizationMachine
- FeaturesLinear
- FieldAwareFactorizationMachine
- InnerProductNetwork
- SENet
- Tower
- LinearBnReluDropBlock ConvBnReluBlock


### normal Layers
Below is a comprehensive list of the 16 layers implemented in this project:

‌Input Layer‌ - Handles the input data.
‌Embedding Layer‌ - Converts categorical features into dense vectors.
‌Fully Connected Layer‌ - Standard dense layer for neural networks.
‌Convolutional Layer‌ - Used for feature extraction in sequential data.
‌Recurrent Layer‌ - Captures temporal dependencies in sequences.
‌Attention Layer‌ - Focuses on important parts of the input data.
‌Batch Normalization Layer‌ - Normalizes the inputs of each layer.
‌Dropout Layer‌ - Prevents overfitting by randomly dropping units.
‌Activation Layer‌ - Applies activation functions (ReLU, sigmoid, tanh, etc.).
‌Pooling Layer‌ - Reduces the dimensionality of the data.
‌Flattening Layer‌ - Converts multi-dimensional data to one-dimensional.
‌Reshape Layer‌ - Reshapes the data to a specified shape.
‌Concatenation Layer‌ - Concatenates multiple inputs.
‌Elementwise Layer‌ - Performs elementwise operations.
‌Residual Layer‌ - Adds skip connections to facilitate gradient flow.
‌Mixture of Experts (MoE) Layer‌ - Utilizes multiple experts to improve model capacity and generalization.
Models
Here is a detailed list of the 23 models available in this project:

‌Collaborative Filtering Model‌ - Uses user-item interactions for recommendations.
‌Content-Based Model‌ - Leverages item content features for recommendations.
‌Hybrid Model‌ - Combines collaborative and content-based approaches.
‌Autoencoder Model‌ - Uses dimensionality reduction for latent feature extraction.
‌Neural Collaborative Filtering (NCF) Model‌ - Deep learning-based collaborative filtering.
‌Factorization Machines (FM) Model‌ - Extends linear models by incorporating factorized interactions.
‌Wide & Deep Model‌ - Combines a linear model (wide) with a deep neural network.
‌DeepFM Model‌ - Integrates FM with deep learning for better feature interactions.
‌Attention-based Model‌ - Employs attention mechanisms to focus on relevant parts of the input.
‌Sequential Recommendation Model‌ - Handles sequential user behavior data.
‌Graph Neural Network (GNN) Model‌ - Utilizes graph structures for recommendations.
‌Matrix Factorization Model‌ - Traditional matrix factorization techniques.
‌Neighborhood-based Model‌ - Recommends items based on user similarity.
‌Latent Dirichlet Allocation (LDA) Model‌ - Topic modeling for content-based recommendations.
‌Variational Autoencoder (VAE) Model‌ - Generative model for latent feature learning.
‌Generative Adversarial Network (GAN) Model‌ - For implicit feedback recommendations.
‌Memory Network Model‌ - Incorporates external memory for better reasoning.
‌Transformer Model‌ - Self-attention mechanism for sequence modeling.
‌BERT-based Model‌ - Bidirectional encoder representations for recommendations.
‌Temporal Convolutional Network (TCN) Model‌ - For time-series recommendation tasks.
‌Hierarchical Attention Network (HAN) Model‌ - Hierarchical attention for document-level recommendations.
‌Convolutional Neural Network (CNN) Model‌ - For image-based recommendations.
‌Recurrent Neural Network (RNN) Model‌ - For sequential data handling in recommendations.
Usage
To use the ‌storch-recommend‌ project, follow these steps:

‌Clone the Repository‌:

bash
Copy Code
git clone https://github.com/mullerhai/storch-recommend.git
cd storch-recommend
‌Build the Project‌:
Ensure you have Scala and SBT (Simple Build Tool) installed. Then, build the project using SBT:

bash
Copy Code
sbt compile
‌Import Layers and Models‌:
In your Scala code, import the necessary layers and models from the torch.recommend package.

‌Configure and Train Models‌:
Configure the models with your dataset and training parameters, then train them using your preferred deep learning library (e.g., PyTorch wrapped in Scala).

‌Evaluate and Deploy‌:
Evaluate the trained models using appropriate metrics and deploy them in your recommendation system.

Contributions
We welcome contributions to this project. Whether it's adding new layers, models, or improving existing ones, your participation is highly appreciated. Please refer to the contribution guidelines for more information.

## Conclusion
This recommendation system provides a rich set of models and layers to meet different recommendation needs. You can choose the appropriate model and layer according to your specific requirements.

### License
This project is licensed under the MIT License.

This README serves as an introductory guide to the ‌storch-recommend‌ project. For more detailed information, refer to the source code and documentation within the repository. Happy recommending!
