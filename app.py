import gradio as gr
import os

from midi2audio import FluidSynth

from utilities.midi import parse_midi, midi_to_wav

def handle_midi(midi_file):
    print(midi_file)
    if not midi_file.name.endswith(".mid"):
        return "Please upload a MIDI file."
    
    notes, durations = parse_midi(midi_file)

    text = str(notes) + str(durations)
    
    return text

def preview_midi(midi_file):
    # Ensure the MIDI file exists
    if not os.path.exists(midi_file):
        gr.Error("MIDI file not found.")
        return "Error: MIDI file not found."

    # Convert MIDI to WAV using FluidSynth
    output_audio = "output.wav"
    gr.Error(output_audio)
    FluidSynth().midi_to_audio(midi_file, output_audio)

    return output_audio

demo = gr.Interface(
    fn=handle_midi,
    inputs=["file"],
    outputs=["text"],
    title="Music Composer with LSTM",
)

if __name__ == "__main__":
    demo.launch()