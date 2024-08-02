Hello welcome to the Sign Language Translator.
This project is based on multiple papers that are trying to solve the communication problems between natural language and sign language.

In this project you can find 3 important modules. I will make just a short presentation but a better one will be added in the future.
1. SequenceToSequenceModel where you can find all the code with which you can build you own Encoder-Decoder RNN based model. You can choose what types of RNN cells to include in your structure and what type of mechanisms would you like to have (Teacher-Forcing, Bahdanau Attention, Scheduler, etc.)
2. Seq2SeqFinetunning where you can find all you need to pick Transformers that have an Encoder-Decoder Architecture and finetunne them with the Sign Language dataset. You will also find error checking methods, and logging systems to help you out during the training loops.
3. GlossToPoseModel - this part contains the image processing part where you can generate the Pose images, concatenate and interpolate them to get a final video/ gif with the translated text.

   Example of a work flow:

   Text: Guten Tag!
   
   Generation: 
![GUT_TAG__trim_shoulderNorm](https://github.com/user-attachments/assets/e5441cf8-7b3e-482f-9fe3-350b3017d7ff)
