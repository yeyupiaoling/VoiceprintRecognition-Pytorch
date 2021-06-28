# 前言
本章介绍如何使用Pytorch实现简单的声纹识别模型，首先你需要熟悉音频分类，没有了解的可以查看这篇文章[《基于PaddlePaddle实现声音分类》](https://blog.doiduoyi.com/articles/1587999549174.html) 。基于这个知识基础之上，我们训练一个声纹识别模型，通过这个模型我们可以识别说话的人是谁，可以应用在一些需要音频验证的项目。

使用环境：

 - Python 3.7
 - Pytorch 1.8.1

# 模型下载
| 数据集 | 准确率 | 下载地址 |
| :---: | :---: | :---: |
| [中文语音语料数据集](https://github.com/KuangDD/zhvoice) | 训练中 | [训练中]() |

# 安装环境
最简单的方式就是使用pip命令安装，如下：
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

注意：libsora和pyaudio容易安装出错，这里介绍解决办法。


libsora安装失败解决办法，使用源码安装，下载源码：[https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/) ，windows的可以下载zip压缩包，方便解压。
```shell
pip install pytest-runner
tar xzf librosa-<版本号>.tar.gz 或者 unzip librosa-<版本号>.tar.gz
cd librosa-<版本号>/
python setup.py install
```

如果出现`libsndfile64bit.dll': error 0x7e`错误，请指定安装版本0.6.3，如`pip install librosa==0.6.3`

安装ffmpeg， 下载地址：[http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/) ，笔者下载的是64位，static版。
然后到C盘，笔者解压，修改文件名为`ffmpeg`，存放在`C:\Program Files\`目录下，并添加环境变量`C:\Program Files\ffmpeg\bin`

最后修改源码，路径为`C:\Python3.7\Lib\site-packages\audioread\ffdec.py`，修改32行代码，如下：
```python
COMMANDS = ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', 'avconv')
```

pyaudio安装失败解决办法，在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)


# 创建数据
本教程笔者使用的是[中文语音语料数据集](https://github.com/KuangDD/zhvoice) ，这个数据集一共有3242个人的语音数据，有1130000+条语音数据。如果读者有其他更好的数据集，可以混合在一起使用，但要用python的工具模块aukit处理音频，降噪和去除静音。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，语音分类标签是指说话人的唯一ID，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中。

在`create_data.py`写下以下代码，因为[中文语音语料数据集](https://github.com/KuangDD/zhvoice) 这个数据集是mp3格式的，作者发现这种格式读取速度很慢，所以笔者把全部的mp3格式的音频转换为wav格式。
```python
def get_data_list(infodata_path, list_path, zhvoice_path):
    with open(infodata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    sound_sum = 0
    speakers = []
    speakers_dict = {}
    for line in tqdm(lines):
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
        save_path = "%s.wav" % sound_path[:-4]
        if not os.path.exists(save_path):
            try:
                wav = AudioSegment.from_mp3(sound_path)
                wav.export(save_path, format="wav")
                os.remove(sound_path)
            except Exception as e:
                print('数据出错：%s, 信息：%s' % (sound_path, e))
                continue
        if sound_sum % 200 == 0:
            f_test.write('%s\t%d\n' % (save_path.replace('\\', '/'), label))
        else:
            f_train.write('%s\t%d\n' % (save_path.replace('\\', '/'), label))
        sound_sum += 1

    f_test.close()
    f_train.close()

if __name__ == '__main__':
    get_data_list('dataset/zhvoice/text/infodata.json', 'dataset', 'dataset/zhvoice')
```

在创建数据列表之后，可能有些数据的是错误的，所以我们要检查一下，将错误的数据删除。
```python
def remove_error_audio(data_list_path):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines1 = []
    for line in tqdm(lines):
        audio_path, _ = line.split('\t')
        try:
            spec_mag = load_audio(audio_path)
            lines1.append(line)
        except Exception as e:
            print(audio_path)
            print(e)
    with open(data_list_path, 'w', encoding='utf-8') as f:
        for line in lines1:
            f.write(line)


if __name__ == '__main__':
    remove_error_audio('dataset/train_list.txt')
    remove_error_audio('dataset/test_list.txt')
```

执行程序，生成数据列表。
```shell
python create_data.py
```

# 数据读取
有了上面创建的数据列表和均值标准值，就可以用于训练读取。主要是把语音数据转换短时傅里叶变换的幅度谱，使用librosa可以很方便计算音频的特征，如梅尔频谱的API为`librosa.feature.melspectrogram()`，输出的是numpy值，可以直接用PaddlePaddle训练和预测。跟梅尔频谱同样很重要的梅尔倒谱（MFCCs）更多用于语音识别中，对应的API为`librosa.feature.mfcc()`。在本项目中使用的API分别是`librosa.stft()`和`librosa.magphase()`。在训练时，使用了数据增强，如随机翻转拼接，随机裁剪。经过处理，最终得到一个`257*257`的短时傅里叶变换的幅度谱。
```python
def load_audio(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    # 推理的数据要移除静音部分
    if mode == 'infer':
        wav = remove_silence(wav, sr)
        wav = remove_noise(wav, sr)
    # 数据拼接
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])
    # 计算短时傅里叶变换
    linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    linear_T = linear.T
    mag, _ = librosa.magphase(linear_T)
    mag_T = mag.T
    freq, freq_time = mag_T.shape
    assert freq_time >= spec_len, "非静音部分长度不能低于1.3s"
    if mode == 'train':
        # 随机裁剪
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag_T[:, rand_time:rand_time + spec_len]
    else:
        spec_mag = mag_T[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, :]
    return spec_mag

# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, model='train', spec_len=257):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode=self.model, spec_len=self.spec_len)
        return spec_mag, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
```


# 训练模型
创建`train.py`开始训练模型，使用的是卷积神经网络模型，本项目提供了`resnet18`、`resnet34`、`resnet50`、`resnet101`、`resnet152`，数据输入层设置为`[None, 1, 257, 257]`，这个大小就是短时傅里叶变换的幅度谱的shape，如果读者使用了其他的语音长度，也需要修改这个值。
```python
def train(args):
    if dist.get_rank() == 0:
        # 日志记录器
        writer = LogWriter(logdir='log')
    # 设置支持多卡训练
    dist.init_parallel_env()
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train', spec_len=input_shape[2])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = CustomDataset(args.test_list_path, model='test', spec_len=input_shape[2])
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 获取模型
    model = resnet50(num_classes=args.num_classes)
    if dist.get_rank() == 0:
        paddle.summary(model, input_size=[(None, ) + input_shape])
    # 设置支持多卡训练
    model = paddle.DataParallel(model)

    # 设置优化方法
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=args.learning_rate,
                                      weight_decay=paddle.regularizer.L2Decay(1e-4))

    # 加载预训练模型
    if args.pretrained_model is not None:
        model.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'optimizer.pdopt')))

    # 获取损失函数
    loss = nn.CrossEntropyLoss()
    train_step = 0
    test_step = 0
    # 开始训练
    for epoch in range(args.num_epoch):
        loss_sum = []
        for batch_id, (spec_mag, label) in enumerate(train_loader()):
            out, feature = model(spec_mag)
            # 计算损失值
            los = loss(out, label)
            loss_sum.append(los)
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and dist.get_rank() == 0:
                print('[%s] Train epoch %d, batch_id: %d, loss: %f' % (
                    datetime.now(), epoch, batch_id, sum(loss_sum) / len(loss_sum)))
                writer.add_scalar('Train loss', los, train_step)
                train_step += 1
                loss_sum = []
        # 多卡训练只使用一个进程执行评估和保存模型
        if dist.get_rank() == 0:
            acc = test(model, test_loader)
            print('[%s] Train epoch %d, accuracy: %f' % (datetime.now(), epoch, acc))
            writer.add_scalar('Test acc', acc, test_step)
            test_step += 1
            save_model(args, model, optimizer)
```

每训练一轮结束之后，执行一次模型评估，计算模型的准确率，以观察模型的收敛情况。
```python
def test(model, test_loader):
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader()):
        label = paddle.reshape(label, shape=(-1, 1))
        out, _ = model(spec_mag)
        acc = accuracy(input=out, label=label)
        accuracies.append(acc.numpy()[0])
    model.train()
    return float(sum(accuracies) / len(accuracies))
```

同样的，每一轮训练结束保存一次模型，分别保存了可以恢复训练的模型参数，也可以作为预训练模型参数。还保存预测模型，用于之后预测。
```python
def save_model(args, model, optimizer):
    input_shape = eval(args.input_shape)
    if not os.path.exists(os.path.join(args.save_model, 'params')):
        os.makedirs(os.path.join(args.save_model, 'params'))
    if not os.path.exists(os.path.join(args.save_model, 'infer')):
        os.makedirs(os.path.join(args.save_model, 'infer'))
    # 保存模型参数
    paddle.save(model.state_dict(), os.path.join(args.save_model, 'params/model.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(args.save_model, 'params/optimizer.pdopt'))
    # 保存预测模型
    paddle.jit.save(layer=model,
                    path=os.path.join(args.save_model, 'infer/model'),
                    input_spec=[InputSpec(shape=(None, ) + input_shape, dtype='float32')])
```

训练过程中，会使用VisualDL保存训练日志，通过启动VisualDL可以随时查看训练结果，启动命令`visualdl --logdir=log --host 0.0.0.0`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504214736429.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504214736432.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504214736389.png)



# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，编写`infer()`函数，在编写模型的时候，模型是有两个输出的，第一个是模型的分类输出，第二个是音频特征输出。所以在这里要输出的是音频的特征值，有了音频的特征值就可以做声纹识别了。我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值`threshold`，读者可以根据自己项目的准确度要求进行修改。
```python
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path1',      str,    'audio/a_1.wav',          '预测第一个音频')
add_arg('audio_path2',      str,    'audio/a_2.wav',          '预测第二个音频')
add_arg('threshold',        float,   0.7,                     '判断是否为同一个人的阈值')
add_arg('input_shape',      str,    '(1, 257, 257)',          '数据输入的形状')
add_arg('model_path',       str,    'models/infer/model',     '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

# 加载模型
model = paddle.jit.load(args.model_path)
model.eval()


# 预测音频
def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    # 执行预测
    _, feature = model(data)
    return feature


if __name__ == '__main__':
    infer(args)
    # 要预测的两个人的音频文件
    feature1 = infer(args.audio_path1)
    feature2 = infer(args.audio_path2)
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > args.threshold:
        print("%s 和 %s 为同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
```


# 声纹识别
在上面的声纹对比的基础上，我们创建`infer_recognition.py`实现声纹识别。同样是使用上面声纹对比的`infer()`预测函数，通过这两个同样获取语音的特征数据。
```python

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('input_shape',      str,    '(1, 257, 257)',          '数据输入的形状')
add_arg('threshold',        float,   0.7,                     '判断是否为同一个人的阈值')
add_arg('model_path',       str,    'models/infer/model',     '预测模型的路径')
args = parser.parse_args()

print_arguments(args)


model = paddle.jit.load(args.model_path)
model.eval()

person_feature = []
person_name = []


def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    # 执行预测
    _, feature = model(data)
    return feature
```

不同的是笔者增加了`load_audio_db()`和`recognition()`，第一个函数是加载语音库中的语音数据，这些音频就是相当于已经注册的用户，他们注册的语音数据会存放在这里，如果有用户需要通过声纹登录，就需要拿到用户的语音和语音库中的语音进行声纹对比，如果对比成功，那就相当于登录成功并且获取用户注册时的信息数据。完成识别的主要在`recognition()`函数中，这个函数就是将输入的语音和语音库中的语音一一对比。
```python
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)


def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro
```


有了上面的声纹识别的函数，读者可以根据自己项目的需求完成声纹识别的方式，例如笔者下面提供的是通过录音来完成声纹识别。首先必须要加载语音库中的语音，语音库文件夹为`audio_db`，然后用户回车后录音3秒钟，然后程序会自动录音，并使用录音到的音频进行声纹识别，去匹配语音库中的语音，获取用户的信息。通过这样方式，读者也可以修改成通过服务请求的方式完成声纹识别，例如提供一个API供APP调用，用户在APP上通过声纹登录时，把录音到的语音发送到后端完成声纹识别，再把结果返回给APP，前提是用户已经使用语音注册，并成功把语音数据存放在`audio_db`文件夹中。
```python
if __name__ == '__main__':
    load_audio_db('audio_db')
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "dataset/temp.wav"

    # 打开录音
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while True:
        try:
            i = input("按下回车键开机录音，录音3秒中：")
            print("开始录音......")
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("录音已结束!")

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 识别对比音频库的音频
            name, p = recognition(WAVE_OUTPUT_FILENAME)
            if p > args.threshold:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        except:
            pass
```
