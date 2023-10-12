import torch, pdb
import torch.nn as nn
#import torchvision.transforms as T
from transformers import AutoModelForSequenceClassification

#class VideoModel(nn.Module):
class VideoModel(nn.Module):
    def __init__(self, model_name='slowfast_r50', pretrained=True, demo=False) -> None:
        super().__init__()

        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        self.model._modules['blocks']._modules['6'].proj = nn.Linear(in_features=2304, out_features=200, bias=True)
        self.demo = demo
        #self.t = T.Resize()

    def forward(self, x):
        if not self.demo:
            x = [elem.squeeze(0) for elem in x]
        
        pred = self.model(x)
        return pred

#class SpectrogramModel(nn.Module):
class SpectrogramModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, demo=False):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrained)
        self.model._modules['fc'] = nn.Linear(in_features=512, out_features=200)
        self.demo = demo

    def forward(self, x):
        if self.demo:
            x = x.unsqueeze(0)
        return self.model(x)


class LanguageModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", output_attentions=True, demo=False):
        """
            Description: Language model which takes input as processed_speech from dataset I have defined and gives the final attention layers as output
            @param model_name: Pretrained model name
            @param output_attentions: Boolean specifies whether to give attention layer output or not
        """
        super(LanguageModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, output_attentions=output_attentions, num_labels=200, ignore_mismatched_sizes=True
        )
        self.demo = demo
        

    def forward(self, tokenized_text):
        """
            Description: Forward function takes the text tokenized by the bert encoder and passes through the model

            @param tokenized_text: Text tokenized using BERT
        """
        if not self.demo:
            tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
            tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
        #print('\n', tokenized_text['input_ids'].size())
        tokenized_text['input_ids'] = tokenized_text['input_ids'][:, :512]
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'][:, :512]

        x = self.model(**tokenized_text).logits
        return x

class UnifiedModel(nn.Module):
    def __init__(self, in_dims=None, intermediate_dims=None, LanguageModel_obj=None, VideModel_obj=None, SpectrogramModel_obj=None):
    #def __init__(self, in_dims=None, intermediate_dims=None, VideModel_obj=None):
    #def __init__(self, in_dims=None, intermediate_dims=None, LanguageModel_obj=None):
    #def __init__(self, in_dims=None, intermediate_dims=None, SpectrogramModel_obj=None):
        """
            Description: A unified model that takes language model output , video_classifier output and audio_classifier output. Here audio_classifier output is spectrogram

            @param in_dims: The dimensions obtained from concatenating language model output , video_classifier output and audio_classifier output
            @param intermediate_dim: The dimension obtained by using an intermediate linear layer over the input obtained from the 'in_dims' layer
            @param LanguageModel_obj: The pytorch model of LanguageModel defined above
            @param VideModel_obj: The pytorch model of VideoModel defined above
            @param SpectrogramModel_obj: The pytorch model of SpectrogramModel defined above
            
        """
        super(UnifiedModel, self).__init__()
        self.in_dims = in_dims #dim_lang_model + dim_video_classifier + dim_audio_classifier
        self.intermediate_dims = intermediate_dims #obtained after linear layer on in_dims
        self.num_classes = 2
        self.LanguageModel_obj = LanguageModel_obj
        self.VideModel_obj = VideModel_obj
        self.SpectrogramModel_obj = SpectrogramModel_obj
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(self.in_dims, self.intermediate_dims)
        self.linear2 = nn.Linear(self.intermediate_dims, self.num_classes)

    def forward(self, language_model_in, video_classifier_in, audio_classifier_in):
    #def forward(self, video_classifier_in):#, audio_classifier_in):
    #def forward(self, language_model_in):#, audio_classifier_in):
    #def forward(self, audio_classifier_in):
        """
            Description: Forward function takes language model output , video_classifier output and audio_classifier output

            @param language_model_in: the processed tokenized input from dataset class
            @param video_classifier_in: the processed video input from dataset class
            @param audio_classifier_in: the processed audio input from dataset class
        """

        language_model_out = self.LanguageModel_obj(language_model_in)
        video_classifier_out = self.VideModel_obj(video_classifier_in)
        #x = self.VideModel_obj(video_classifier_in)
        #x = self.LanguageModel_obj(language_model_in)
        audio_classifier_out = self.SpectrogramModel_obj(audio_classifier_in)
        #x = self.SpectrogramModel_obj(audio_classifier_in)
        x = torch.cat((language_model_out, video_classifier_out, audio_classifier_out), axis=-1)
        #x = torch.cat((language_model_out, video_classifier_out), axis=-1)
        x = self.linear1(x)
        x = self.relu1(x) 
        x = self.linear2(x)
        #x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    llm = LanguageModel()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    inputs = tokenizer(text, return_tensors="pt")    

    with torch.no_grad():
        out_ = llm(inputs)
        print(out_.size())


