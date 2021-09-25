# How do Transformers work?

## A bit of Transformer history

![trtrtr](https://user-images.githubusercontent.com/59636424/134766745-13afce2d-7701-42ef-9ebc-8da954efcdc7.PNG)

> * GPT (2018.6)
> > 다양한 NLP task를 수행하며 fine-tuning을 위해 사용되는 첫번째 사전학습 transformer 모델로 SOTA모델로 성과를 얻음
> * BERT (2018.10)
> > 문장 요약에 더 좋은 성과를 보여준 큰 사전학습 모델이다.
> * GPT-2 (2019.2)
> > 좀 더 커진 GPT 버전!
> * DistilBERT (2019.10)
> > BERT의 성능 97%를 보유하지만 메모리적으로 40% 가벼워지고 60% 더 빨라진 BERT의 distilled version!
> * BART & T5 (2019.10)
> > 원래 Transformer 모델로서 같은 구조로 사용되는 2개의 큰 사전학습 모델
> * GPT-3 (2020.5)
> > fine-tuning없이 다양한 task를 잘 수행하는 GPT-2의 큰 버전! (zero-shot learning)


## Transformers are language models

* **Self-supervised learning은 data에 label을 붙일 필요가 없으므로** 자동으로 계산되는 훈련이다!

사전학습 모델은 transfer learning(전이 학습) 과정을 거친다! -> 특정 task에 대해 유용하게 하기 위해서!!

전이학습 과정 중에 supervised 방법으로 fine_tuning을 하는데 직접 labeling한 label을 이용한다!

* casual language modeling(인과 언어 모델링)

앞의 단어 n개를 읽어 문장의 다음 단어를 예측하는 방식으로 미래 input에 따라 output이 달라지지 않는다!

![xxxx](https://user-images.githubusercontent.com/59636424/134768079-f94f6206-dcf1-4a63-96d9-ea63d8c850e5.PNG)

* masked language modeling

mask된 단어를 예측한다!

![maskmask](https://user-images.githubusercontent.com/59636424/134768108-8098af91-c92b-4a20-8bce-afab44c070d3.PNG)

## Transformers are big models

사전학습된 데이터양 뿐만 아니라 모델 사이즈도 증가시킴으로써 더 좋은 성과를 내는 전략으로 흘러가고 있다!(DistillBERT와 달리)

![qwqwqw](https://user-images.githubusercontent.com/59636424/134768260-ae633d00-d46c-4102-8df2-7afcab0f51b3.PNG)

하지만 이렇게 흘러갈수록 많은 계산할 수 있는 자원과 시간을 요구한다!

**그러므로 훈련된 weights들을 공유하는 것이 전반적은 계산 비용을 줄일 수 있다.**

## Transfer Learning

pretrain -> fine-tuning 순으로 진행!

* 그러면 final task를 위해 직접 훈련하는 방법은 어떨까?

  1. 이미 pretrain된 데이터 세트와 유사한 데이터 세트에 대해 교육되었으므로 굳이 직접 훈련할 필요는 없다.

  2. 이미 많은 데이터로 train되었기에 많이 적은 데이터로 fine-tuning을 하는 것이 적절한 효과를 얻을 것이다.

  3. 필요한 시간과 자원이 훨씬 더 적다.

* example

English로 사전 훈련된 모델을 활용 -> arXiv 말뭉치에서 fine-tuning하여 science/research 기반 모델 생성! (이것을 transfer learning이라 한다!)

---

**최대한 task에 근접한 데이터로 pretrain된 모델을 활용해 fine-tuning을 하는 것이 좋다!**


## General architecture

transformer 구조를 알아보자!

### Introduction

* Encoder: 입력을 수신하여 representation을 구축한다. -> input에 대한 이해를 얻으려 한다.

* Decoder: target을 만들기 위해서 encoder의 representation과 다른 input을 이용한다. output을 생성하는 역할을 한다.

![endecoder](https://user-images.githubusercontent.com/59636424/134770412-dafb88d2-48fd-438f-87f7-ef9a5c015366.PNG)

* Encoder-only models: 문장 분류(SC)와 객체 인식(NER)와 같이 input의 understanding을 필요로 하는 task에 적합

* Decoder-only models: 문장 생성(text generation)와 같은 생성 task에 적합

* Encoder-decoder models or squence-to-squence models: 번역 또는 요약와 같은 생성 task에 적합

### Attention layers

자연어 처리에서 해당 단어를 앞과 뒤 단어들로 맥락에 의해 영향을 받는 것을 살펴보는 layer이다.

### The original architecture

Transformer 구조는 원래 번역(translation)을 위해서 디자인되었다.

* encoder

특정 언어로 input을 받는다.

문장의 모든 단어 사용

* decoder

원하는 대상 언어로 동일한 문장 수신!

순차적으로 작동하며 timestep t의 단어를 번역하면 t 이전의 단어(이미 번역한 문장 단어)들만 사용가능하다!


![ddd](https://user-images.githubusercontent.com/59636424/134771343-19a503cd-dbc6-42b5-ab19-d97818bcd6cc.PNG)


Decoder의 첫 번째 attention layer는 모든 input에 주의를 기울린다. (mask를 통해 일부 단어를 가린다!)

Decoder의 두 번째 attention layer는 encoder의 output을 사용한다. -> 현재 단어 예측을 가장 잘하기 위해서 모든 input 문장을 접근한다.

이것은 다른 언어들의 다른 문장 순서 & 다른 문법을 가지기에 유용! => 최선의 번역!

### Architectures vs checkpoints

* Architecture: 모델의 뼈대

* Checkpoints: 주어진 구조에 load되는 가중치

* Model: 모호한 말이지만 architecture이나 checkpoints를 의미한다.


