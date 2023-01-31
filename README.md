# Classifying-false-alarm
광운대학교 산학연계 프로젝트 - 머신러닝 기반의 SW 정적시험 거짓 경보 자동 분류 시스템 개발

## Introduction
SW 신뢰성 시험은 SW가 동작할 수 있는 다양한 경우의 수를 확인함으로써 SW가 일으킬 수 있는 결함을 식별하는 정적 시험, 동적 시험을 의미하며, 대부분의 기업에서 개발 과정에서 사용하는 정적 분석 도구로 소스 코드를 검사한다면 프로그램을 실행하지 않은 상태에서 잠재적인 취약점을 검출하여 SW의 결함을 파악할 수 있다. 

하지만, 시스템의 대규모 및 높은 복잡성으로 인해 완벽하게 정확한 정적 분석이 불가능하기 때문에 실제 문제를 놓치지 않도록 과도하게 근사하여 오류가 없지만 오류가 있다고 하는 거짓 경보(false alarm)가 많이 발생한다. 이러한 거짓 경보를 전문가들이 수작업으로 다시 분류하다 보니 시간이 오래 걸린다는 문제가 있다. 

따라서, 문제 해결을 위해 Programming Language 데이터로 인공지능을 학습시켜 간단하고 빠르게 거짓 경보를 분류하는 모델을 개발하면, 보다 더 고품질의, 여러 비용을 최소화하는 효율적인 SW를 개발할 수 있다.

## 주요 기능
정적 분석 도구를 통해 오류라고 발생한 C언어 코드의 함수 부분을 입력하면, 해당 오류가 false positive인지, true positive인지 분류한다.

## Getting Started
1. repository clone
2. https://drive.google.com/drive/folders/1BkqbZemq9QiauJZGLfQPWwhrg_xh_2Ns 에서 models 폴더 다운로드 후, 압축 파일 내 models 폴더를 해당 폴더로 이동
> 폴더 용량 이슈로 google drive를 통해 다운로드
3. pip install -r requirements.txt 를 통해 필요한 모듈 설치
4. python app.py 명령어로 프로젝트를 실행 후, http://localhost:7860/ 접속

또는, https://huggingface.co/spaces/minseokKoo/Auto_Classifier 에 접속하여 테스트

## 실행 화면
<img width="1154" alt="1" src="https://user-images.githubusercontent.com/67617479/215723739-77d323d4-caca-4133-b9c1-6e205e856ff1.png">

## Requirements
- numpy
- pandas
- torch
- transformers
- tensorflow-cpu
- sentencepiece
- gradio
