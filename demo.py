import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
#import IPython.display as ipd
import sys
sys.path.append('waveglow/')
import numpy as np
import torch
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from scipy.io import wavfile
from datetime import datetime

t1 = datetime.now()
img_folder = "inference_mels/"
audio_folder = "inference_wavs/"

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')
        fig.savefig(img_folder+'test.png')

# Setup hparams
hparams = create_hparams()
hparams.sampling_rate = 22050

# Load the tacotron2 model
checkpoint_path = "models/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# Load the waveglow model
waveglow_path = 'models/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)
t2 = datetime.now()
delta = t2 - t1
print("Load models only: ", delta.total_seconds())

t3 = datetime.now()
# Make a sentence to synthesize
text = "Hi! it's Jennifer here. And this is myvoice a.i."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()


# Decode the text and plot the mel-spectrogram results
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
#plot_data((mel_outputs.float().data.cpu().numpy()[0],
#           mel_outputs_postnet.float().data.cpu().numpy()[0],
#           alignments.float().data.cpu().numpy()[0].T))


# Synthesize the specgtrogram using Waveglow
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

# Do it again but remove the Waveglow bias
audio_denoised = denoiser(audio, strength=0.01)[:, 0]
y = audio_denoised.cpu().numpy()
sr = hparams.sampling_rate
wavfile.write(audio_folder+"waveglow_no_bias.wav", sr, y.T)
t4 = datetime.now()
delta = t4 - t3
print("Sentence to write wavefile: ", delta.total_seconds())
