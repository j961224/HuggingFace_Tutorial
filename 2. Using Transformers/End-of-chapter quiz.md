# End-of-chapter quiz

## 1. 언어 모델 pipeline의 순서는?

1. tokenizer로 text를 다루고 ID들을 반환! =>> tokenizing

2. model은 ID들을 다루고 output을 예측!

3. tokenizer는 예측값을 text로 다시 변환시켜준다! => de-tokenizing

## 2. Transformer model의 tensor output 차원은?

batch size x sequence length x hidden size

## 3. subword tokenization 예시는?

WordPiece, BPE, Unigram

## 4. model head는?

여러 개의 layer를 구성된 요소로 transformer 예측을 task별로 변환!

## 5. AutoModel은?

AutoModel은 초기화할 checkpoint만 알면 된다!

## 6. 다른 길이의 batching sequence가 같이 있으면?

padding 사용해서 길이 맞춰줌!

truncating으로 잘라서 맞춰줌!

Attention masking으로 적용하지 않을 token을 가려줌!

## 7. sequence 분류 모델의 logit output에 softmax를 적용한 point는?

total 값이 1은 값으로 변환시켜주고 확률적 값으로 보여준다!

## 9. code를 통해 결과값은?

~~~
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
~~~

![cc](https://user-images.githubusercontent.com/59636424/134801442-596a5ffd-1cb0-4ea9-a19d-64c0a48ebe16.PNG)

각 string을 token화 시켜준다!

## 10. code에 틀린 부분은?

~~~
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
~~~

tokenizer를 만들 당시에는 bert를 썼는데 model은 gpt2 사전학습모델을 가져옴으로써 같은 checkpoint를 쓰지 않았다!

