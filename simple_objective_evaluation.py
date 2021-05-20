import argparse
import os
from hparams import hparams as hparamspy
from tacotron.models import create_model
import tensorflow as tf
from tacotron.feeder import Feeder
from tacotron.feeder_reference import FeederReference
from datasets import audio
import subprocess
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
import pandas as pd

import numpy as np
from datasets.audio import *
import os
from hparams import hparams

def MCD(x, y):
    assert len(x) == len(y) != 0
    outer_sum = 0
    for t in range(len(x)):
        inner_sum = 0
        for dim in range(1, 60):
        # for dim in range(1, 25):
            inner_sum += math.pow((x[t][dim] - y[t][dim]), 2)
        outer_sum += math.sqrt(inner_sum)
    alpha = (10*math.sqrt(2))/math.log(10)
    print("Debug : outer_sum={}".format(outer_sum))
    print("Debug : alpha={}".format(alpha))
    print("Debug : 1/len(x)={}".format(1 / len(x)))
    return (alpha/len(x))*outer_sum


def spec2wav(linears, hparams, save_path):
    wav = audio.inv_mel_spectrogram(linears.T, hparams)
    audio.save_wav(wav, save_path, sr=hparams.sample_rate)


class MgcHandler:

    def __init__(self, alpha, mcsize, nFFTHalf, sampling_frequency):
        self.alpha = alpha
        self.mcsize = mcsize
        self.nFFTHalf = nFFTHalf
        self.sampling_frequency = sampling_frequency
        # self.speech_tool_kit="/vrac/asini/workspace/merlin_install/merlin/tools/bin/SPTK-3.9/"

    @staticmethod
    def world_analysis(output_folder, basename,world_vocoder_path, debug=False):
        # Spectral analysis using WORLD vocoder
        cmd = "{}./analysis {} {} {} {}".format(world_vocoder_path,os.path.join(output_folder, basename + ".wav"),
                                                  os.path.join(output_folder, basename + ".f0"),
                                                  os.path.join(output_folder, basename + ".sp"),
                                                  os.path.join(output_folder, basename + ".bapd"))
        if debug:
            print(cmd)
        subprocess.run(cmd.split())

    def sp2mgc(self, output_folder, basename):
        # Conversion of the cepstral coefficients into MGC
        x2x = subprocess.Popen(('x2x', '+df', os.path.join(output_folder, basename + ".sp")), stdout=subprocess.PIPE)
        sopr = subprocess.Popen(('sopr', '-R', '-m', '32768.0'), stdin=x2x.stdout, stdout=subprocess.PIPE)
        x2x.stdout.close()
        with open(os.path.join(output_folder, basename + ".mgc"), "w") as mgc_file:
            mcep = subprocess.Popen(
                ('mcep', '-a', str(self.alpha), '-m', str(self.mcsize), '-l', str(self.nFFTHalf), '-e', '1.0E-8', '-j',
                 '0', '-f', '0.0', '-q', '3'), stdin=sopr.stdout, stdout=mgc_file)
            sopr.stdout.close()
            mcep.wait()

    def mgc2sp(self, output_folder, basename):
        mgc2sp = subprocess.Popen(('mgc2sp', '-a', str(self.alpha), '-g', '0', '-m', str(self.mcsize), '-l', str(self.nFFTHalf), '-o', '2', os.path.join(output_folder, basename + ".mgc")), stdout=subprocess.PIPE)
        sopr = subprocess.Popen(('sopr', '-d', '32768.0', '-P'), stdin=mgc2sp.stdout, stdout=subprocess.PIPE)
        mgc2sp.stdout.close()
        with open(os.path.join(output_folder, basename + ".sp"), "w") as sp_file:
            x2x = subprocess.Popen(('x2x', '+fd'), stdin=sopr.stdout, stdout=sp_file)
            sopr.stdout.close()
            x2x.wait()

    @staticmethod
    def mgc2mgca(output_folder, basename):
        # Conversion of the MGC file from float/binary to string
        with open(os.path.join(output_folder, basename + ".mgca"), "w") as mgc_file:
            x2x = subprocess.Popen(('x2x', '+fa', os.path.join(output_folder, basename + ".mgc")), stdout=mgc_file)
            x2x.wait()

    @staticmethod
    def mgca2mgc(output_folder, basename):
        # Conversion of the MGC file from string to float/binary
        with open(os.path.join(output_folder, basename + ".mgc"), "w") as mgc_file:
            x2x = subprocess.Popen(('x2x', '+af', os.path.join(output_folder, basename + ".mgca")), stdout=mgc_file)
            x2x.wait()

    def f02f0a(self, output_folder, basename):
        # Conversion of the F0 file from int/binary to string
        with open(os.path.join(output_folder, basename + ".f0a"), "w") as f0_file:
            x2x = subprocess.Popen(('x2x', '+da', os.path.join(output_folder, basename + ".f0")), stdout=f0_file)
            x2x.wait()

    def f0a2f0(self, output_folder, basename):
        # Conversion of the F0 file from string to int/binary
        with open(os.path.join(output_folder, basename + ".f0"), "w") as f0_file:
            x2x = subprocess.Popen(('x2x', '+ad', os.path.join(output_folder, basename + ".f0a")), stdout=f0_file)
            x2x.wait()

    @staticmethod
    def __read_from_Xa(filename):
        # Returns a list containing the wanted features
        vector = []
        with open(filename, "r") as file:
            for line in file:
                vector.append(float(line.strip()))
        return np.array(vector)

    def read_mgca(self, output_folder, basename):
        # Returns a list containing the MGC features
        mgc_vector = self.__read_from_Xa(os.path.join(output_folder, basename + ".mgca"))
        assert len(mgc_vector) % (self.mcsize + 1) == 0
        mgc_vector = mgc_vector.reshape((-1, self.mcsize + 1))
        return mgc_vector

    def read_f0a(self, output_folder, basename, debug=False):
        # Returns a list containing the MGC features
        f0_vector = self.__read_from_Xa(os.path.join(output_folder, basename + ".f0a"))
        f0_vector = f0_vector.reshape((-1, 1))
        return f0_vector

    @staticmethod
    def __write_to_Xa(filename, vector):
        with open(filename, "w") as file:
            a = np.ravel(vector)
            for j in a:
                file.write(str(j) + "\n")

    def write_mgca(self, output_folder, basename, mgc_vector):
        self.__write_to_Xa(os.path.join(output_folder, basename + ".mgca"), mgc_vector)

    def write_f0a(self, output_folder, basename, f0_vector):
        self.__write_to_Xa(os.path.join(output_folder, basename + ".f0a"), f0_vector)

    def feature_extraction(self, output_folder, basename):
        self.world_analysis(output_folder, basename)
        self.sp2mgc(output_folder, basename)
        self.mgc2mgca(output_folder, basename)
        mgc_vector = self.read_mgca(output_folder, basename)
        self.f02f0a(output_folder, basename)
        f0_vector = self.read_f0a(output_folder, basename)

        return mgc_vector, f0_vector

    def synthesis(self, output_folder, filename, mgc_vector, f0_vector, bapd_filename):
        self.write_mgca(output_folder, filename, mgc_vector)
        self.mgca2mgc(output_folder, filename)
        self.write_f0a(output_folder, filename, f0_vector)
        self.f0a2f0(output_folder, filename)
        self.mgc2sp(output_folder, filename)
        synth = subprocess.Popen(('world_synth', str(self.nFFTHalf), str(self.sampling_frequency),
                                  os.path.join(output_folder, filename + ".f0"),
                                  os.path.join(output_folder, filename + ".sp"),
                                  os.path.join(output_folder, bapd_filename + ".bapd"),
                                  os.path.join(output_folder, filename + ".wav")))
        synth.wait()



def build_args():
    parser=argparse.ArgumentParser(description='')
    # parser.add_argument('input_path', type=str, nargs=1, help='data input')
    parser.add_argument('linears_pred', type=str, nargs=1, help='data input')
    parser.add_argument('linears_true', type=str, nargs=1, help='data input')
    parser.add_argument('output_folder',type=str, nargs=1, help='data input')
    return parser.parse_args()


def main():
    args=build_args()


    # Prepare results DataFrame
    # df = pd.read_csv(args.input_path, delimiter="|", header=None)
    # df["MCD"] = np.NaN
    # df["MCDbis"] = np.NaN
    # if debug_flag:
    #     print(df.head())
    output_folder=args.output_folder[0]
    linears_pred = np.load(args.linears_pred[0])
    linears_true = np.load(args.linears_true[0])
    print(linears_pred.shape,linears_true.shape)

    # # Take off the batch wise padding
    target_lengths = 9999
    linears_pred = linears_pred[:target_lengths]
    linears_true = linears_true[:target_lengths]

    # # Go back to wav
    spec2wav(linears_pred, hparams, os.path.join(output_folder, 'tmp_pred_{}.wav'.format("aud")))
    spec2wav(linears_true, hparams, os.path.join(output_folder, 'tmp_true_{}.wav'.format("aud")))

    mgc_handler = MgcHandler(nFFTHalf=1024, alpha=0.58, mcsize=59, sampling_frequency=48000)
    # # Extract MGC
    mgc_pred, f0_pred = mgc_handler.feature_extraction(output_folder, 'tmp_pred_aud')
    mgc_true, f0_true = mgc_handler.feature_extraction(output_folder, 'tmp_true_aud')

    # # Find alignment with DTW
    # if debug_flag:
    #     print(mgc_pred.shape, mgc_true.shape)
    # distance, path = fastdtw(mgc_pred, mgc_true, dist=euclidean)
    # if debug_flag:
    #     print(distance, len(path))

    # # Align the MGC and F0 according to the DTW path
    # aligned_pred_mgc, aligned_true_mgc = [], []
    # aligned_pred_f0, aligned_true_f0 = [], []
    # for j in range(len(path)):
    #     alignment_j = path[j]
    #     aligned_pred_mgc.append(mgc_pred[alignment_j[0]])
    #     aligned_true_mgc.append(mgc_true[alignment_j[1]])
    #     aligned_pred_f0.append(f0_pred[alignment_j[0]])
    #     aligned_true_f0.append(f0_true[alignment_j[1]])
    # if debug_flag:
    #     print(len(aligned_pred_mgc), len(aligned_true_mgc))
    #     print(len(aligned_pred_f0), len(aligned_true_f0))

    # # Try to resynthesize
    # mgc_handler.synthesis(output_folder, 'tmp_pred_aligned_{}'.format(i), aligned_pred_mgc, aligned_pred_f0, 'tmp_pred_{}'.format(i))
    # mgc_handler.synthesis(output_folder, 'tmp_true_aligned_{}'.format(i), aligned_true_mgc, aligned_true_f0, 'tmp_true_{}'.format(i))

    # # Compute MCD two different way
    # mcd = MCD(aligned_pred_mgc, aligned_true_mgc)
    # mcd_bis = -1
    # cdist = subprocess.Popen(('cdist', '-m', str(mgc_handler.mcsize+1),
    #                           os.path.join(output_folder, 'tmp_pred_aligned_{}'.format(i)+".mgc"),
    #                           os.path.join(output_folder, 'tmp_true_aligned_{}'.format(i)+".mgc"),
    #                           '-o', '0'), stdout=subprocess.PIPE)
    # x2x = subprocess.Popen(('x2x', '+fa'), stdin=cdist.stdout, stdout=subprocess.PIPE)
    # mcd_bis = x2x.stdout.read()
    # cdist.stdout.close()
    # mcd_bis = float(mcd_bis.strip())

    # if verbose_flag:

    #     print("MCD = {}".format(mcd))
    #     print("MCDbis = {}".format(mcd_bis))
    #     print("=================")

    # # Save result
    # df.loc[i, "MCD"] = mcd
    # df.loc[i, "MCDbis"] = mcd_bis
    # df.to_csv(os.path.join(output_folder, "results.csv"), sep="|")

    # # Remove temporary file
    # for base in ["tmp_pred_", "tmp_pred_aligned_", "tmp_true_", "tmp_true_aligned_"]:
    #     filename = base+str(i)
    #     filename = os.path.join(output_folder, base+str(i))
    #     for ext in [".bapd", ".f0", ".f0a", ".mgc", ".mgca", ".sp"]:
    #         try:
    #             os.remove(filename+ext)
    #         except:
    #             pass



if __name__ == '__main__':
    main()