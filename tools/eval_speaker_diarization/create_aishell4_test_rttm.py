import os

import soundfile
from pyannote.database.util import load_rttm
from tqdm import tqdm
from yeaudio.audio import AudioSegment


def create_rttm(annotation_dir, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_w:
        for file in os.listdir(annotation_dir):
            if not file.endswith(".rttm"): continue
            with open(os.path.join(annotation_dir, file), 'r', encoding='utf-8') as f_r:
                lines = f_r.readlines()
                for line in lines:
                    f_w.write(line)


def create_audio_path_list(audio_dir, list_path):
    with open(list_path, 'w', encoding='utf-8') as f_w:
        for file in os.listdir(audio_dir):
            if not file.endswith(".flac"): continue
            file_path = os.path.join(audio_dir, file).replace('\\', '/')
            name = file.split('.')[0]
            f_w.write(f'{file_path}\t{name}\n')


def create_audio_db(data_list_path, rttm_path, output_dir):
    annotations = load_rttm(rttm_path)
    with open(data_list_path, 'r') as f_r:
        for line in tqdm(f_r.readlines(), desc='裁剪说话人音频'):
            audio_path, name = line.strip().split('\t')
            audio_segment = AudioSegment.from_file(audio_path)
            sample_rate = audio_segment.sample_rate
            audio = audio_segment.samples
            annotation = annotations[name]
            for segment, track, label in annotation.itertracks(yield_label=True):
                if segment.end - segment.start < 0.3: continue
                save_path = os.path.join(output_dir, name, label, f'{track}.wav')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                audio_sub = audio[int(segment.start * sample_rate):int(segment.end * sample_rate)]
                soundfile.write(save_path, audio_sub, sample_rate)


if __name__ == '__main__':
    create_rttm(annotation_dir='dataset/test/TextGrid', output_path='dataset/references.rttm')
    create_audio_path_list(audio_dir='dataset/test/wav', list_path='dataset/data_list.txt')
    create_audio_db(data_list_path='dataset/data_list.txt', rttm_path='dataset/references.rttm',
                    output_dir='dataset/audio_db/')
