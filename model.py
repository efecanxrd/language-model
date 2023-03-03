import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

# define the model architecture
model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_size),
    LSTM(units=128, return_sequences=True),
    LSTM(units=128),
    Dense(units=vocab_size, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_data, epochs=num_epochs, validation_data=val_data)

# generate text using the trained model
generated_text = generate_text(model, starting_text, num_chars_to_generate)

# helper function for generating text
def generate_text(model, starting_text, num_chars_to_generate):
    # preprocess the starting text
    input_text = preprocess_text(starting_text)
    
    # initialize the generated text
    generated_text = input_text
    
    # generate the remaining characters one by one
    for i in range(num_chars_to_generate):
        # convert the generated text to a sequence of token IDs
        input_seq = [token_to_id[token] for token in generated_text]
        
        # pad the sequence to a fixed length
        input_seq = pad_sequences([input_seq], maxlen=max_seq_length-1)
        
        # predict the next token
        predicted_probs = model.predict(input_seq)[0]
        predicted_id = tf.random.categorical(predicted_probs, num_samples=1)[-1,0].numpy()
        
        # convert the predicted token ID to a character
        predicted_token = id_to_token[predicted_id]
        
        # append the predicted character to the generated text
        generated_text += predicted_token
    
    return generated_text

# helper function for preprocessing text
def preprocess_text(text):
    # convert the text to lowercase
    text = text.lower()
    
    # remove non-alphabetic characters
    text = re.sub(r'[^a-z ]', '', text)
    
    # tokenize the text
    tokens = text.split()
    
    return tokens
