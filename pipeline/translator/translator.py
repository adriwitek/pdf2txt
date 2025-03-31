''' Adrian Rubio @ BSC 2025
    The purpose of this script is to translate the parquets of NP25 with salamadraTA
    
    '''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import pandas as pd
import argparse
from datetime import datetime
import json
from translator.texttokenizer import TextTokenizer
from translator.lang_identificator import *



# MACROS
NUM_BEAMS=5
EARLY_STOPPING=True




def _lang_code_translator(src_lang):
   
    translated_lang_code = ''
    if( 'cat' in src_lang or 'ca' in src_lang):
      translated_lang_code = 'Catalan'
    elif( 'glg' in src_lang or 'gl' in src_lang):
        translated_lang_code = 'Galician'
    elif( 'eus' in src_lang or 'eu' in src_lang):
        translated_lang_code = 'Default'
    else:
      raise Exception('Translation language couldnt be located.')
   
    return translated_lang_code



def _lang_code_translator_2(src_lang): # quick test..
    '''MT Spanish, Euskera, Galician,Catalan'''
    translated_lang_code = ''
    if( 'cat' in src_lang or 'ca' in src_lang):
      translated_lang_code = 'Catalan'
    elif( 'glg' in src_lang or 'gl' in src_lang):
        translated_lang_code = 'Galician'
    elif( 'eus' in src_lang or 'eu' in src_lang):
        translated_lang_code = 'Euskera'
    else:
      raise Exception('Translation language couldnt be located.')
   
    return translated_lang_code


def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def _apply_prompt_template(x,src_lang_code, tgt_lang_code):
  return f'[{src_lang_code}] {x} \n[{tgt_lang_code}]'


def _get_lenght(input_ids, output_index):
  return input_ids[output_index].shape[0]


def _detokenize(input_ids, outputs, tokenizer):
  '''Aux. functionf of _translate_sentences'''
  with torch.no_grad():
    results_detokenized = [ tokenizer.decode(output[_get_lenght(input_ids, i):], skip_special_tokens=True).strip() for i, output in enumerate(outputs) ]
  
  return results_detokenized # list ---> TODO: CHECK THIS OUT NOW THAT IS A BATCH


def _translate_sentences(prompts, tokenizer,model, max_sentence_lenght):
  '''promts: a list of strings'''

  with torch.no_grad():
    #tokenize
    encodings = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=True)
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)
    #mt
    #with torch.cuda.amp.autocast():  # Enable mixed precision (FP16) inference
    with torch.amp.autocast('cuda'):
      outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=NUM_BEAMS,max_length=max_sentence_lenght,early_stopping=EARLY_STOPPING)

  #detokenize
  results_detokenized = _detokenize(input_ids, outputs, tokenizer)



  return results_detokenized



def _preprocess_document(lang,text):
    '''Return sentences ready to be translated'''
    # Maybe if a read the lang from the parquet i can skip the fastparquet part

    if(lang is None or lang == ''):   # detect  
      lang = get_language(text)
    try:
      lang_code =  _lang_code_translator(lang)
    except:
      lang_code = 'Catalan'
    tokenizer_splt = TextTokenizer(lang_code )
    sentences_all, translate = tokenizer_splt.tokenize(text)
    try:
      src_lang_code =  _lang_code_translator_2(lang) # e.g. Catalan
    except:
      src_lang_code = 'Catalan'
    tgt_lang_code = 'Spanish'
    sentences = [_apply_prompt_template(s,src_lang_code, tgt_lang_code) for s, t in zip(sentences_all, translate) if t == True ]
    return sentences



def translate_doc(original_doc_name,lang, text,  tokenizer, model, batch_size = 50, max_sentence_lenght=500):
  
  sentences = _preprocess_document(lang,text) #splits depending on lang
  batched_generator = batch_generator(sentences, batch_size)

  i=0
  translated = []
  for batch in batched_generator:
      try:
        result = _translate_sentences(batch, tokenizer,model, max_sentence_lenght)
      except Exception as e:
        logging.debug(f'err while processing batch in doc{original_doc_name}, skiping.... Exception:\n{e}')
        result = []
      translated.extend(result)
      i+=1

  return '\n'.join(translated)



def _read_df(df):
  for row in df.itertuples(index=False):
    lang =  row.lang
    if(lang == 'es'):
      logging.debug(f'Skipping es doc: { row.original_doc_name}')
    elif(lang == 'ca' or lang == 'gl' or lang == 'eu'):
      yield row.procurement_id, row.original_doc_name, lang, row.content
    else:
      pass



def _read_df_all(df):
  for row in df.itertuples(index=False):
      yield row.procurement_id, row.original_doc_name, row.lang, row.content






def init_models(    model_id = '/gpfs/projects/bsc88/NextProcurement/Pipeline/pipeline/pdf2txt/salamadratagguf',
                    gguf_model_name= 'salamandraTA-2B.Q2_K.gguf',
                    max_sentence_lenght=500,
                    batch_size = 50,
                    verbose = True
                ):


  if(verbose):
      logging.getLogger().setLevel(logging.DEBUG)
  logging.debug(f'model_id:{model_id}')
  logging.debug(f'gguf_model_name:{gguf_model_name}')
  logging.debug(f'batch_size:{batch_size}')
  logging.debug(f'max_sentence_lenght:{max_sentence_lenght}')


  # Load model and move to GPU
  logging.debug(f'Loading MT models...')
  torch_dtype = torch.bfloat16  # could be torch.float16 or torch.bfloat16 too
  tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left') #gguf_file=gguf_model_name,
  if(gguf_model_name is None or gguf_model_name == 'args.gguf_model_name'):
    logging.debug(f'Discarding the use of GGUF')
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation='eager', torch_dtype=torch_dtype)
  else:
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation='eager',  gguf_file=gguf_model_name, torch_dtype=torch_dtype)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.debug(f'device{device}')
  model = model.to(device)
  
  return model, tokenizer
  
  
  
  #translated_text = translate_doc(original_doc_name,lang, content, args.batch_size, tokenizer,model, args.max_sentence_lenght)
    
 






 








