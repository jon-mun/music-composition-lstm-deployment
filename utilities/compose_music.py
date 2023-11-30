import os
import pickle
import numpy as np
from keras.utils import to_categorical

import keras.backend as K 
from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, Reshape
from keras.layers import Flatten, RepeatVector, Permute, TimeDistributed
from keras.layers import Multiply, Lambda, Softmax
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.models import load_model
from keras.optimizers.legacy import RMSprop

# utilities
from utilities.midi import parse_midi, midi_to_wav
from music21 import note, stream, instrument, chord, duration
import time

run_folder = os.path.join(os.getcwd(), 'utilities')
store_folder = os.path.join(run_folder, 'store')
output_folder = os.path.join(run_folder, 'output')

# Parameters
seq_len = 32

def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)

def sample_with_temp(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

def prepare_sequences(notes, durations, lookups, distincts, seq_len =32):
    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, duration_names, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]
    return (network_input, network_output)

def prepare_network(
    notes, durations,
):
    if os.path.exists(os.path.join(store_folder, 'distincts')):
        with open(os.path.join(store_folder, 'distincts'), 'rb') as f:
            distincts = pickle.load(f)
    else:
        print(f'No distincts found in {store_folder}')

    if os.path.exists(os.path.join(store_folder, 'lookups')):
        with open(os.path.join(store_folder, 'lookups'), 'rb') as f:
            lookups = pickle.load(f)
    else:
        print(f'No lookups found in {store_folder}')

    # prepare input for transfer learning
    network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len)

    return (network_input, network_output, distincts, lookups)

def create_network(n_notes, n_durations, embed_size = 100, rnn_units = 256, use_attention = False):
    notes_in = Input(shape = (None,))
    durations_in = Input(shape = (None,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in) 
    x = Concatenate()([x1,x2])
    x = LSTM(rnn_units, return_sequences=True)(x)

    if use_attention:
        x = LSTM(rnn_units, return_sequences=True)(x)
        e = Dense(1, activation='tanh')(x)
        e = Reshape([-1])(e)
        alpha = Activation('softmax')(e)
        alpha_repeated = Permute([2, 1])(RepeatVector(rnn_units)(alpha))
        c = Multiply()([x, alpha_repeated])
        c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_units,))(c)    
    else:
        c = LSTM(rnn_units)(x)
                                    
    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)
   
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    if use_attention:
        att_model = Model([notes_in, durations_in], alpha)
    else:
        att_model = None
        
    opti = RMSprop(lr = 0.001)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, att_model

def init_model(n_notes, n_durations):
    # initialize model
    model, att_model = create_network(n_notes, n_durations, use_attention=True)

    # print model summary
    model.summary()

    # load weights
    weights_folder = os.path.join(run_folder, 'weights')
    model.load_weights(os.path.join(weights_folder, "weights.h5"))

    return (model, att_model)
    
def create_callbacks():
    early_stopping = EarlyStopping(
        monitor='loss'
        , restore_best_weights=True
        , patience = 10
    )

    callbacks_list = [
        early_stopping
    ]

    return callbacks_list

def predict(model, notes, durations, lookups):
    # prediction params
    notes_temp=0.5
    duration_temp = 0.5
    max_extra_notes = 200
    max_seq_len = 32
    seq_len = 32

    notes = ['START']
    durations = [0]

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups

    if seq_len is not None:
        notes = ['START'] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    prediction_output = []
    notes_input_sequence = []
    durations_input_sequence = []
    overall_preds = []

    for n, d in zip(notes,durations):
        note_int = note_to_int[n] ##sticky note: part ini menandakan input yang masuk ke dalam model untuk dipredict nextnya apa. Ganti ke data baru/lagu lain
        duration_int = duration_to_int[d]## sticky note: retrieve elemen ke n (terakhir) dari data train
        
        notes_input_sequence.append(note_int)
        durations_input_sequence.append(duration_int)
        
        prediction_output.append([n, d])

    for note_index in range(max_extra_notes):
        prediction_input = [
            np.array([notes_input_sequence])
            , np.array([durations_input_sequence])
        ]

        notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)

        new_note = np.zeros(128)
        
        for idx, n_i in enumerate(notes_prediction[0]):
            try:
                note_name = int_to_note[idx]
                midi_note = note.Note(note_name)
                new_note[midi_note.pitch.midi] = n_i            
            except:
                pass
            
        overall_preds.append(new_note)            
        
        i1 = sample_with_temp(notes_prediction[0], notes_temp)
        i2 = sample_with_temp(durations_prediction[0], duration_temp)    

        note_result = int_to_note[i1]
        duration_result = int_to_duration[i2]
        
        prediction_output.append([note_result, duration_result])

        notes_input_sequence.append(i1)
        durations_input_sequence.append(i2)
        
        if len(notes_input_sequence) > max_seq_len:
            notes_input_sequence = notes_input_sequence[1:]
            durations_input_sequence = durations_input_sequence[1:]
            
        if note_result == 'START':
            break

    return prediction_output

def prediction_to_midi(prediction_output):
    output_folder = os.path.join(run_folder, 'output')

    midi_stream = stream.Stream()

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        note_pattern, duration_pattern = pattern
        # pattern is a chord
        if ('.' in note_pattern):
            notes_in_chord = note_pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Violoncello()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)
        elif note_pattern == 'rest':
        # pattern is a rest
            new_note = note.Rest()
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            midi_stream.append(new_note)
        elif note_pattern != 'START':
        # pattern is a note
            new_note = note.Note(note_pattern)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            midi_stream.append(new_note)

    midi_stream = midi_stream.chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_file = 'output-' + timestr + '.mid'
    midi_stream.write('midi', fp=os.path.join(output_folder, new_file))

    return new_file

def compose_music(midi_file):
    # preprocess input music
    notes, durations = parse_midi(midi_file)

    # prepare sequences used to train the Neural Network
    network_input, network_output, distincts, lookups = prepare_network(notes, durations)

    # create model
    n_notes = 23948
    n_durations = 32

    model, att_model = init_model(n_notes, n_durations)

    # callbacks
    callbacks_list = create_callbacks()

    # fit model
    # model.fit(network_input, network_output
    #       , epochs=25, batch_size=32
    #       , validation_split = 0.2
    #       , callbacks=callbacks_list
    #       , shuffle=True
    #      )
    
    # # predict
    prediction_output = predict(model, notes, durations, lookups)

    # # convert prediction to midi
    new_midi_file = prediction_to_midi(prediction_output)

    # # midi to wav
    wav_file = midi_to_wav(new_midi_file)

    return wav_file

    
    
    



    