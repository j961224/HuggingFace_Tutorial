# Sequence-to-sequence models

encoder-decoder 모델은 Transformer 구조 두 부분을 사용한다.

encoder는 모든 단어 접근이 가능하지만, decoder는 timestep t 이전에 단어들만 접근 가능하다!

ex) T5는 임시 text 범위를 single mask로 바꿔 pretrain 되어진다. -> 그리고 mask 예측

**요약, 번역 or 생성 QnA와 같이 주어진 입력에 따라 새로운 문장 생성 task에 적합!**

## Model 예시

* BART
* mBART
* Marian
* T5
