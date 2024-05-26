# HW3
**Install
**\
사이트를 통해 파이토치 설치\
1.사양확인(https://pytorch.org/get-started/locally/)\
2.cuda download 검색 후, Toolkit을 다운로드하는데 아까 확인했던 사양으로 다운.\
3.cuda 설치\
4.cudnn 검색 후, Archived cuDNN Releases에서 해당되는 사양으로 다운.\
5.다운받은 cuDNN파일안에 있는 파일을 아까 설치한 cuda 파일 안에 옮김.(위치는 다를 수 있으나, 기본 위치는 드라이브-Programs Files-NVIDIA GPU Computing Toolkit)\
6.Anaconda 검색 후, 최신버전으로 설치\
7.Anaconda Prompt 실핼 후, $ conda create -n torch python=3.9 입력 후 설치.\
8.$ conda activate torch 으로 가상환경을 킴.\
9.(1)에서 사양을 확인하며 적혀있던 Run this Command를 입력.\
10.$ python을 실행하여 파이썬 인터프리터를 킴.\
11. >>> import torch 입력 후, >>>torch.cuda.is_available()을 입력했을 때, True가 나온다면 정상적으로 설치된 것.\
12. 다운받은 파일의 source파일의 경로로 이동.\
13. python main.py를 통해 파일 실행하여 학습가능.\
\
Moudlue error의 경우, pip install 'error'(<-모들이름)을 통해 설치가능.\

