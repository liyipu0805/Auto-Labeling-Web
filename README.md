# Web端图像自动批量标注软件

# 一. 功能介绍

1. 支持图像批量上传

2. 支持图像批量自动标注

3. 支持批量导出标准文件（.xml）

4. 支持容器化部署

# 二. 所需环境

# 三. 部署方法

## 1. 基于本地环境

适用环境：Linux系统（Ubuntu、Centos）或 Windows

安装运行所必要的库

```bash
pip install -r requirements.txt
```

在根目录输入

```bash
python app.py
```

根据提示打开网页 **localhost:8080** 即可完成部署

## 2. 基于Docker

适用环境：Linux系统（Ubuntu、Centos）

### a. 基于Dockerfile

在根目录输入

```bash
docker build -t auto_label_web .
```

等待镜像安装完成后，输入

```bash
docker run -d -p 5000:8080 auto_label_web
```

访问 **localhost:5000** 即可完成部署

### b. 基于网络镜像

通过docker拉取网络镜像

```bash
docker pull nicejeo/auto-label-web:latest
```

启动容器

```bash
docker run -d -p 5000:8080 nicejeo/auto-label-web
```

访问 **localhost:5000** 即可完成部署

# 四. 使用方法

1. 为方便标注，在本地创建文件夹，命名为**JPEGImages**，将待标注的图片放入该文件夹。

2. 在Web Demo中选择文件夹中的所有图片，点击“上传”按钮，等待图像上传、标注完毕。

3. 点击“全部导出”按钮，将以压缩包的形式下载所有图像的标注，解压缩到文件夹**labels**中。

4. 打开**labelimg**软件，先点击“**change save dir**”按钮，选择**labels**文件夹，再点击“**open dir**”按钮，选择**JPEGImages**文件夹，软件中即可加载原始图像及其对应的自动标注结果，用户可以根据需求进行微调。
   
   注：labelimg软件可以通过命令行获取
   
   ```bash
   pip install labelimg
   labelimg
   ```

至此，Web端图像自动批量标注软件的说明结束。
