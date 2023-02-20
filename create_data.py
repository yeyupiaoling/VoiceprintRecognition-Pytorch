import json
import os
import time
from multiprocessing import Pool, cpu_count
from datetime import timedelta

from pydub import AudioSegment


# 生成数据列表
def get_data_list(infodata_path, zhvoice_path):
    print('正在读取标注文件...')
    with open(infodata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    speakers = []
    speakers_dict = {}
    for line in lines:
        line = json.loads(line.replace('\n', ''))
        duration_ms = line['duration_ms']
        if duration_ms < 1300:
            continue
        speaker = line['speaker']
        if speaker not in speakers:
            speakers_dict[speaker] = len(speakers)
            speakers.append(speaker)
        label = speakers_dict[speaker]
        sound_path = os.path.join(zhvoice_path, line['index'])
        data.append([sound_path.replace('\\', '/'), label])
    print(f'一共有{len(data)}条数据！')
    return data


def mp32wav(num, data_list):
    start = time.time()
    for i, data in enumerate(data_list):
        sound_path, label = data
        if os.path.exists(sound_path):
            save_path = sound_path.replace('.mp3', '.wav')
            if not os.path.exists(save_path):
                wav = AudioSegment.from_mp3(sound_path)
                wav.export(save_path, format="wav")
                os.remove(sound_path)
        if i % 100 == 0:
            eta_sec = ((time.time() - start) / 100 * (len(data_list) - i))
            start = time.time()
            eta_str = str(timedelta(seconds=int(eta_sec)))
            print(f'进程{num}进度：[{i}/{len(data_list)}]，剩余时间：{eta_str}')


def split_data(list_temp, n):
    length = len(list_temp) // n
    for i in range(0, len(list_temp), length):
        yield list_temp[i:i + length]


def main(infodata_path, list_path, zhvoice_path, to_wav=True, num_workers=2):
    data_all = []
    data = get_data_list(infodata_path=infodata_path, zhvoice_path=zhvoice_path)
    if to_wav:
        print('准备把MP3总成WAV格式...')
        split_d = split_data(data, num_workers)
        pool = Pool(num_workers)
        for i, d in enumerate(split_d):
            pool.apply_async(mp32wav, (i, d))
        pool.close()
        pool.join()
        for d in data:
            sound_path, label = d
            sound_path = sound_path.replace('.mp3', '.wav')
            if os.path.exists(sound_path):
                data_all.append([sound_path, label])
    else:
        for d in data:
            sound_path, label = d
            if os.path.exists(sound_path):
                data_all.append(d)
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')
    for i, d in enumerate(data_all):
        sound_path, label = d
        if i % 200 == 0:
            f_test.write(f'{sound_path}\t{label}\n')
        else:
            f_train.write(f'{sound_path}\t{label}\n')
    f_test.close()
    f_train.close()


if __name__ == '__main__':
    main(infodata_path='dataset/zhvoice/text/infodata.json',
         list_path='dataset',
         zhvoice_path='dataset/zhvoice',
         to_wav=True,
         num_workers=cpu_count())
