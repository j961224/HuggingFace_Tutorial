# End-of-chapter quiz

## 1. Twitter message emotions의 label 중, basic emotions이 아닌 것은?

confunsion

## 2. Hub에 ar_sarcasm dataset에 task는?

ar_sarcasm dataset은 텍스트 분류, 감정 분류 task에 활용 가능!

## 3. BERT model의 문장 쌍은?

[CLS] Tokens_of_sentence_1 [SEP] Tokens_of_sentence_2 [SEP]

## 4. Dataset.map의 이익은?

함수 결과가 cached되어 재실행되지 않는다!

dataset의 각 요소에 함수를 적용하는 것보다 더 빠르게 multi-processing을 적용 가능!

메모리에 전체 Dataset을 load하지 않으므로 하나의 요소가 가능한한 처리되어 결과에 즉시 저장!

## 5. dynamic padding 의미는?

batch가 만들어지고, batch 내에 max length를 맞춰준다.

## 6. collate function 목적은?

batch에 모든 샘플들을 묶어준다. -> DataCollatorWithPadding function을 사용하여 모든 item을 1번에 패딩하여 길이가 같도록 해준다!

## 7. train된 것과 다른 task인 사전학습모델(bert-base-uncased)의 AutoModelForXxxclasses 중 하나를 instantiate할 때 일어나는 것은?

사전학습모델의 head를 버리고 task에 적절한 새로운 head로 넣어준다!

## 8. TrainingArguments의 목적은?

Trainer로 training과 evaluation에 사용되는 모든 hyperparameters를 포함한다.

(TrainingArguments("test-trainer")를 사용하며 test, train 둘다 적용!)

## 9. Accelerate library를 왜 사용하니?

multiple GPUs와 TPUs로 training loops를 작동할 수 있도록 하기 위해서!
