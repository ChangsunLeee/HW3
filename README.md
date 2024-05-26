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

**result**
처음 코드를 실행했을 때, 과적합(Overfitting)되는 경향이 생겼음.\
![Figure_1](https://github.com/ChangsunLeee/HW3/assets/167077784/b935bcd1-c14d-4e51-9a4b-8aa3fb0dfcb8)\
처음 Loss값을 받아보았을 땐, 줄어들다가 epoch이 증가함에 따라 Loss값이 줄어들지 않고 증가하거나 유지되는 경향을 보임.(epoch 1000)\
이과정을 해소하기 위해, 드롭아웃을 사용하고 학습률을 낮췄으며, 과정을 확인하기 위해 Loss값이 50epoch이상 감소되지 않는다면 조기종료를 시키기로 함.\
1. 드롭아웃을 쓴 이유-랜덤성부여\
2. 학습율을 낮춘 이유-일반화 성능을 높이기 위해(훈련데이터셋에 너무 맞추지 않기 위해)\
3. 조기종료를 하는 이유-훈련에 너무 많은 시간을 소요.\
\
바뀐코드로 돌린 결과\
![Figure_2](https://github.com/ChangsunLeee/HW3/assets/167077784/f28ff995-398c-4cb2-bc30-3a36d8fcb2be)\
시간이 지남에 따라 Loss값도 감소함.
