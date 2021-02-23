# ASR
### End-to-end Automatic Speech Recognition project

#### Note: Project is still under development.

#### Current progress:

+ Initial project planning => completed
  * The LibriSpeech dataset has been selected. This dataset was selected as it has over 600 hours of validated audio clips with transcriptions.
  * Google colab has been selected for training as it provides free access to a GPU.
  * The PyTorch library will be used for the deep learning parts of this project. This library was chosen as it allows for fast prototyping and can be used in production.
  * The problem of automatic speech recognition will be framed as follows: Audio data will be converted from waveforms into Mel Spectrograms, each frame of the Melspectrogram
  will be passed to the model. The model will predict a character associated with each frame.
  * Loss function: As audio clip transcriptions are not time-aligned with the audio clips the Connectionist temporal classification (CTC) loss function will be used. 
  * Prediction decoding: During the early stages of model development greedy decoding will be used for simplicity. After the model has been finalized a CTC-beam search
  will be used to decode the predictions of the model.
  
 + Model selection => completed
  * As this task involves mapping sequences to characters an RNN architecture is a natural choice.
  * RNN's are not well suited to performing feature extraction and will struggle with the raw Melspectrgrams. As a result of this, a CNN will be used to perform feature extraction.
  * To reduce the dimensionality of the CNN outputs an MLP (Multi-layer perceptron) will be used.
  * To map the hidden states to characters at each time step a MLP will be used.
  * CNN -> RNN -> MLP
  
  + Architecture evaluation => completed
    * A large variety of architectures were tested to determine which is most well suited to the task.
    * The models tested can be found in the phase_3 branch of the project currently.
    * The final model that was selected is as follows: 
    2 layer CNN => 2 Layer Residual-CNN => 2 Layer fully-connected => 2 layer unidirectional LSTM  => classifier layer
    
  + Model training => completed
    * Model was trained on the LibriSpeech clean 100 and 360 datasets.
    * Model converged to a good solution given the amount of training.
    * Example model predictions using greedy decoding (worse case):
        - label:    and yesterday things went on just as usual
        - prediction: and yesterday thinks when un just as usual
        - label:    i am very tired of swimming about here o mouse
        - prediction: i am very tard of swimming about here o mouse
        - label:    cried alice again for this time the mouse was bristling all over and she felt certain it must be really offended
        - prediction: cried oursagain for this tin the mouse bas bristling all over and she felt sertain it must be really af fenit
  
  + Transfer learning => in progress
    * As it is unfeasible to create an ASR model capable of functioning well for all speakers using only 600 hours of speech and a single GPU
      we will make use of transfer learning to create user specific models.
    * The aforementioned trained model will be used as the base model when applying transfer learning
    * To get an accurate measurement of how well the system will perform it is necessary to implement the remainder of the features that support the 
      models predictions when in use - Language model, spell checker and the CTC-beam search (all implemented already).
    * Currently I am testing different methods of transfer learning to determine which method balances increase in performance and user experience
      as generating training data can be quite time consuming for users.

           
           
 

        
        
        


       
       
       
       


      
      
    
    
    
    
    
    
    
    
    
    
