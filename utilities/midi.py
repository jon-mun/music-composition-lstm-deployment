from midi2audio import FluidSynth
from music21 import corpus, converter, instrument, note, stream, chord, duration

def parse_midi(midi_file):
    # data params
    intervals = range(1)
    seq_len = 32

    notes = []
    durations = []

    original_score = converter.parse(midi_file).chordify()
    for interval in intervals:
        score = original_score.transpose(interval)

        notes.extend(['START'] * seq_len)
        durations.extend([0]* seq_len)

        for element in score.flatten():
            if isinstance(element, note.Note):
                if element.isRest:
                    notes.append(str(element.name))
                    durations.append(element.duration.quarterLength)
                else:
                    notes.append(str(element.nameWithOctave))
                    durations.append(element.duration.quarterLength)

            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                durations.append(element.duration.quarterLength)

    return notes, durations

def midi_to_wav(midi_file):
    fs = FluidSynth()
    wav_file = midi_file.name.replace(".mid", ".wav")
    fs.midi_to_audio(midi_file, wav_file)
    
    return wav_file