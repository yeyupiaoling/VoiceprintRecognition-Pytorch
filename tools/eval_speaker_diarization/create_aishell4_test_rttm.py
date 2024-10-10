import os


def create_rttm(annotation_dir, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_w:
        for file in os.listdir(annotation_dir):
            if not file.endswith(".rttm"): continue
            with open(os.path.join(annotation_dir, file), 'r', encoding='utf-8') as f_r:
                lines = f_r.readlines()
                speaker_names = []
                for line in lines:
                    name = line.strip().split(' ')[7]
                    if name not in speaker_names:
                        speaker_names.append(name)
                speaker_names_dict = {name: k for k, name in enumerate(speaker_names)}
                for line in lines:
                    data = line.strip().split(' ')
                    data[7] = str(speaker_names_dict[data[7]])
                    f_w.write(' '.join(data) + '\n')


def create_audio_path_list(audio_dir, list_path):
    with open(list_path, 'w', encoding='utf-8') as f_w:
        for file in os.listdir(audio_dir):
            if not file.endswith(".flac"): continue
            file_path = os.path.join(audio_dir, file).replace('\\', '/')
            name = file.split('.')[0]
            f_w.write(f'{file_path}\t{name}\n')


if __name__ == '__main__':
    create_rttm(annotation_dir='dataset/test/TextGrid', output_path='dataset/references.rttm')
    create_audio_path_list(audio_dir='dataset/test/wav', list_path='dataset/data_list.txt')
