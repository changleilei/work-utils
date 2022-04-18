# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/4/15 6:51 下午
==================================="""
import torch
from transformers import AutoTokenizer, \
    DistilBertForSequenceClassification

import pandas as pd
from transformers.modeling_outputs import SequenceClassifierOutput


class DistilBertForMultilabelSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict)

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
              'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
              'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
              'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
print(len(label_cols))

label2id = {key: i for i, key in enumerate(label_cols)}
id2label = {value: key for key, value in label2id.items()}

cuda = 'cuda' if torch.cuda.is_available() else 'cpu'


def functions():
    data_path = '../data/dailydialogue/dailydilogue_qamatch.csv'
    tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-go-emotion")
    model = DistilBertForMultilabelSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-go-emotion").to(cuda)
    frame = pd.read_csv(data_path)
    result = {'questions': [], 'answers': [], 'questions_label': [], 'answers_label': []}
    for i in range(0, len(frame), 30):

        questions = frame['questions'][i:i+30].to_list()
        answers = frame['answers'][i:i+30].to_list()
        questions_embedding = tokenizer(questions, padding=True, return_tensors="pt").to(cuda)
        answers_embedding = tokenizer(answers, padding=True, return_tensors="pt").to(cuda)

        with torch.no_grad():
            question_outputs = model(**questions_embedding)
            question_logits = question_outputs.logits
            question_label_ids = [logit.argmax().item() for logit in question_logits]
            question_labels = [id2label[label_id] for label_id in question_label_ids]

            answer_output = model(**answers_embedding)
            answer_logits = answer_output.logits
            answer_label_ids = [logit.argmax().item() for logit in answer_logits]
            answer_labels = [id2label[label_id] for label_id in answer_label_ids]
            result['questions'].extend(questions)
            result['answers'].extend(answers)
            result['questions_label'].extend(question_labels)
            result['answers_label'].extend(answer_labels)

    pd.DataFrame(result).to_csv('../data/dailydialogue/emotion_dailydialogue.csv')



if __name__ == '__main__':
    functions()