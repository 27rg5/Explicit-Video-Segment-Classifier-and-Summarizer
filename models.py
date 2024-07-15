import torch, pdb
import sk2torch
import torch.nn as nn
#import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MLP(nn.Module):
    def __init__(self, captionnet, weighted_loss_strategy=False):
        super(MLP, self).__init__()

        self.captionnet = sk2torch.wrap(captionnet).to(torch.float)
        self.weighted_loss_strategy = weighted_loss_strategy
        in_features = self.captionnet.module.layers[-1].in_features
        self.captionnet = self.captionnet.module.layers
        for p in self.captionnet.parameters():
            p.requires_grad = False

        if not self.weighted_loss_strategy:
            self.captionnet = self.captionnet[:-1]
            self.captionnet.append(nn.Linear(in_features, in_features))
            self.captionnet.append(nn.ReLU())
    
    def forward(self, x):
        x = self.captionnet(x)
        return x

class VideoModel(nn.Module):
    def __init__(self, model_name='slowfast_r50', pretrained=True, out_embed_dim=200, demo=False) -> None:
        super().__init__()

        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        self.model._modules['blocks']._modules['6'].proj = nn.Linear(in_features=2304, out_features=out_embed_dim, bias=True)
        self.demo = demo
        #self.t = T.Resize()

    def forward(self, x):
        
        if not self.demo:
            x = [elem.squeeze(0) for elem in x]
        
        pred = self.model(x)
        return pred

class SpectrogramModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, out_embed_dim=200, demo=False):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrained)
        self.model._modules['fc'] = nn.Linear(in_features=512, out_features=out_embed_dim)
        self.demo = demo

    def forward(self, x):
        
        if self.demo:
            x = x.unsqueeze(0)
        return self.model(x)


class LanguageModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", output_attentions=True, out_embed_dim=200, demo=False):
        """
            Description: Language model which takes input as processed_speech from dataset I have defined and gives the final attention layers as output
            @param model_name: Pretrained model name
            @param output_attentions: Boolean specifies whether to give attention layer output or not
        """
        super(LanguageModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, output_attentions=output_attentions, num_labels=out_embed_dim, ignore_mismatched_sizes=True
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

class LateFusionWithAttention(nn.Module):
    def __init__(self, hidden_dim, self_attention=False, num_heads=1):
        super(LateFusionWithAttention, self).__init__()
        self.self_attention = self_attention
        self.hidden_dim = hidden_dim 
        self.multiheadattention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, tensor_non_null_dict):
        
        if not self.self_attention:
            modalities_present  = [k for k in tensor_non_null_dict.keys()]
            # Pairwise attention
            if len(modalities_present)==4:
                language_model_out, video_classifier_out, audio_classifier_out, caption_classifier_out = tensor_non_null_dict.values()
                atten1, _ = self.multiheadattention(language_model_out, video_classifier_out, audio_classifier_out)
                atten2, _ = self.multiheadattention(language_model_out, audio_classifier_out, video_classifier_out)
                atten3, _ = self.multiheadattention(audio_classifier_out, video_classifier_out, language_model_out)
                atten4, _ = self.multiheadattention(language_model_out, caption_classifier_out, audio_classifier_out)
                atten5, _ = self.multiheadattention(language_model_out, caption_classifier_out, video_classifier_out)
                atten6, _ = self.multiheadattention(video_classifier_out, caption_classifier_out, audio_classifier_out)
                atten7, _ = self.multiheadattention(video_classifier_out, caption_classifier_out, language_model_out)
                atten8, _ = self.multiheadattention(audio_classifier_out, caption_classifier_out, video_classifier_out)
                atten9, _ = self.multiheadattention(audio_classifier_out, caption_classifier_out, language_model_out)
                atten10, _ = self.multiheadattention(language_model_out, video_classifier_out, caption_classifier_out)
                atten11, _ = self.multiheadattention(video_classifier_out, audio_classifier_out, caption_classifier_out)
                atten12, _ = self.multiheadattention(audio_classifier_out, language_model_out, caption_classifier_out)
                concatenate_list = [atten1, atten2, atten3, atten4, atten5, atten6, atten7, atten8, atten9, atten10, atten11, atten12]
            elif len(modalities_present)==3:
                language_model_out, video_classifier_out, audio_classifier_out = tensor_non_null_dict.values()
                atten1, _ = self.multiheadattention(language_model_out, video_classifier_out, audio_classifier_out)
                atten2, _ = self.multiheadattention(language_model_out, audio_classifier_out, video_classifier_out)
                atten3, _ = self.multiheadattention(audio_classifier_out, video_classifier_out, language_model_out)
                atten4, _ = self.multiheadattention(audio_classifier_out, language_model_out, video_classifier_out)
                atten5, _ = self.multiheadattention(video_classifier_out, language_model_out, audio_classifier_out)
                atten6, _ = self.multiheadattention(video_classifier_out, audio_classifier_out, language_model_out)   
                concatenate_list = [atten1, atten2, atten3, atten4, atten5, atten6]
            concatenated_attention = torch.cat(concatenate_list, dim=-1).squeeze(1)
        else:
            # Concatenate and then self-attention
            concat_modalities = torch.cat(list(tensor_non_null_dict.values()), dim=-1)
            concatenated_attention,_ = self.multiheadattention(concat_modalities, concat_modalities, concat_modalities)
        
        return concatenated_attention

class UnifiedModel(nn.Module):
    def __init__(self, out_dims, intermediate_dims, in_dims, vanilla_fusion=False, self_attention=False, LanguageModel_obj=None, VideModel_obj=None, SpectrogramModel_obj=None, mlp_object=None, weighted_loss_mlp_fusion=False):
        """
            Description: A unified model that takes language model output , video_classifier output and audio_classifier output. Here audio_classifier output is spectrogram

            @param in_dims: The dimensions obtained from concatenating language model output , video_classifier output and audio_classifier output
            @param intermediate_dim: The dimension obtained by using an intermediate linear layer over the input obtained from the 'in_dims' layer
            @param LanguageModel_obj: The pytorch model of LanguageModel defined above
            @param VideModel_obj: The pytorch model of VideoModel defined above
            @param SpectrogramModel_obj: The pytorch model of SpectrogramModel defined above
            
        """
        super(UnifiedModel, self).__init__()
        self.self_attention = self_attention
        self.in_dims = in_dims 
        self.out_dims = out_dims
        self.num_heads = 2
        self.intermediate_dims = intermediate_dims
        self.num_classes = 2
        self.LanguageModel_obj = LanguageModel_obj
        self.VideModel_obj = VideModel_obj
        self.SpectrogramModel_obj = SpectrogramModel_obj
        self.relu1 = nn.ReLU()
        self.vanilla_fusion = vanilla_fusion 
        self.weighted_loss_mlp_fusion = weighted_loss_mlp_fusion
        self.mlp_object = mlp_object
        self.mlp = None

        if not self.vanilla_fusion:
            self.latefusionwithattention = LateFusionWithAttention(self.in_dims, self.self_attention, self.num_heads)
        self.linear1 = nn.Linear(self.out_dims, self.intermediate_dims)
        self.linear2 = nn.Linear(self.intermediate_dims, self.num_classes)

        if self.mlp_object:
            self.mlp = MLP(self.mlp_object, self.weighted_loss_mlp_fusion)


    def forward(self, language_model_in=None, video_classifier_in=None, audio_classifier_in=None, doc_topic_distr_in=None):
        """
            Description: Forward function takes language model output , video_classifier output and audio_classifier output

            @param language_model_in: the processed tokenized input from dataset class
            @param video_classifier_in: the processed video input from dataset class
            @param audio_classifier_in: the processed audio input from dataset class
        """
        language_model_out = self.LanguageModel_obj(language_model_in) if self.LanguageModel_obj else None
        video_classifier_out = self.VideModel_obj(video_classifier_in) if self.VideModel_obj else None
        audio_classifier_out = self.SpectrogramModel_obj(audio_classifier_in) if self.SpectrogramModel_obj else None
        caption_classifier_out = self.mlp(doc_topic_distr_in) if self.mlp else None
        tensor_list = [('language', language_model_out), ('video', video_classifier_out), ('audio', audio_classifier_out), ('caption', caption_classifier_out)]
        tensor_non_null_dict = dict()
        for key, value in tensor_list:
            if value is not None:
                tensor_non_null_dict[key] = value
        
        if not self.vanilla_fusion:
            if not self.self_attention:
                for key, _ in tensor_non_null_dict.items():
                    tensor_non_null_dict[key] = tensor_non_null_dict[key].unsqueeze(1)
            
            x = self.latefusionwithattention(tensor_non_null_dict)
        else:
            x = torch.cat(list(tensor_non_null_dict.values()), dim=-1)


        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        
        return x


if __name__ == '__main__':
    llm = LanguageModel()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    inputs = tokenizer(text, return_tensors="pt")    

    with torch.no_grad():
        out_ = llm(inputs)
        print(out_.size())


