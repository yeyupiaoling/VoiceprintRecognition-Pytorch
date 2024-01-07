import os


# 制作CN-Celeb数据集列表
# 下载地址：https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz
# 下载并解压到dataset目录，解压命令：tar -zxvf cn-celeb_v2.tar.gz
def create_cn_celeb(list_path, data_path='dataset/'):
    f_train = open(list_path, 'w', encoding='utf-8')
    data_dir = os.path.join(data_path, 'CN-Celeb_flac/data/')
    dirs = sorted(os.listdir(data_dir))
    for label, d in enumerate(dirs):
        # 跳过测试集
        if label >= 800:continue
        for file in os.listdir(os.path.join(data_dir, d)):
            sound_path = os.path.join(data_dir, d, file).replace('\\', '/')
            f_train.write(f'{sound_path}\t{label}\n')
    f_train.close()


# 制作CN-Celeb2数据集列表
# 下载分包1地址：https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzaa
# 下载分包2地址：https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzab
# 下载分包3地址：https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzac
# 下载并解压到dataset目录，合并压缩包命令：cat cn-celeb2_v2.tar.gza* > cn-celeb2_v2.tar.gz，解压命令：tar -zxvf cn-celeb2_v2.tar.gz
def create_cn_celeb2(list_path, data_path='dataset/'):
    f_train = open(list_path, 'a', encoding='utf-8')
    data_dir = os.path.join(data_path, 'CN-Celeb2_flac/data/')
    dirs = sorted(os.listdir(data_dir))
    last_label = 800
    for label, d in enumerate(dirs):
        for file in os.listdir(os.path.join(data_dir, d)):
            sound_path = os.path.join(data_dir, d, file).replace('\\', '/')
            f_train.write(f'{sound_path}\t{label + last_label}\n')
    f_train.close()


if __name__ == '__main__':
    create_cn_celeb(list_path='dataset/train_list.txt', data_path='dataset')
    create_cn_celeb2(list_path='dataset/train_list.txt', data_path='dataset')
