import gradio as gr

from utilities.midi import parse_midi, midi_to_wav
from utilities.compose_music import compose_music

def handle_midi(midi_file):
    if not midi_file.name.endswith(".mid"):
        return "Please upload a MIDI file."
    
    wav_file = compose_music(midi_file=midi_file)
    
    return wav_file

with gr.Blocks() as demo:
    gr.Markdown("# Music Composer with LSTM")
    with gr.Row():
        with gr.Column():
            inp = gr.File(label="Upload MIDI file", file_types=["mid"])
            preview = gr.Audio(label="Preview MIDI")

            preview_btn = gr.Button("Preview Midi")
            preview_btn.click(fn=midi_to_wav, inputs=inp, outputs=preview)
        
        out = gr.Audio(label="Generated Music")
    
    btn = gr.Button("Submit")
    btn.click(fn=handle_midi, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()