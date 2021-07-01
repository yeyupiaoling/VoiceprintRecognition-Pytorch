
# libsora安装出错解决办法


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

# pyaudio安装出错解决办法
pyaudio安装失败解决办法，在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)

