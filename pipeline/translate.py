from bs4 import BeautifulSoup
import ctranslate2
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import unicodedata
import os
import argparse
from texttokenizer import TextTokenizer




# Translation Model

TARGET_LANG = 'spa'

#TRANSLATION_MODEL_DIR='/app/pdf2txt/pipeline/models/nllb/nllb-200-distilled-600M/'
TRANSLATION_MODEL_DIR='/app/pdf2txt/pipeline/models/nllb/nllb-ct2/'


TRANSLATION_MODEL_DIR_tok='/app/pdf2txt/pipeline/models/nllb/nllb-200-distilled-600M/'



MAX_SENTENCE_BATCH=40

TARGET_PREFIX = ['spa_Latn']



def _lang_code_translator(src_lang):
   
    translated_lang_code = ''
   
    if( 'cat' in src_lang ):
      translated_lang_code = 'Catalan'
    elif( 'glg' in src_lang):
        translated_lang_code = 'Galician'
    elif( 'eus' in src_lang):
        translated_lang_code = 'Default'
    else:
      raise Exception('Translation language couldnt be located.')
   

    return translated_lang_code


def _normalize_input_string(result):
    result = unicodedata.normalize('NFC', result)
    return result

def _translate_batch(input_batch, spm, model, max_sentence_batch=10):

    batch_input_tokenized = []
    batch_input_markers = []

    #preserve_markup = PreserveMarkup()

    num_sentences = len(input_batch)
    for pos in range(0, num_sentences):
        tokenized = spm.convert_ids_to_tokens(spm.encode(input_batch[pos]))
        batch_input_tokenized.append(tokenized)

    batch_output = []
    for offset in range(0,len(batch_input_tokenized), max_sentence_batch):
      batch = batch_input_tokenized[offset:offset+max_sentence_batch]
      partial_result = model.translate_batch(batch, 
                                            return_scores=False, 
                                            replace_unknowns=True, 
                                            target_prefix=[TARGET_PREFIX]*len(batch))
      for pos in range(0,len(partial_result)):
        tokenized = partial_result[pos][0]['tokens'][1:]
        translated = spm.decode(spm.convert_tokens_to_ids(tokenized))
        #print(translated)
        batch_output.append(translated)

    return batch_output




def translate(text, tokenizer, spm, translator):
  sentences, translate = tokenizer.tokenize(text)
  num_sentences = len(sentences)
  sentences_batch = []
  indexes = []
  results = ["" for x in range(num_sentences)]
  for i in range(num_sentences):
      if translate[i] is False:
          continue

      sentences_batch.append(sentences[i])
      indexes.append(i)

  translated_batch = _translate_batch(sentences_batch, spm, translator, MAX_SENTENCE_BATCH)
  for pos in range(0, len(translated_batch)):
      i = indexes[pos]
      results[i] = translated_batch[pos]

  #Rebuild split sentences
  translated = tokenizer.sentence_from_tokens(sentences, translate, results)
  return translated





def translate_xml_tree(xml_string, tokenizer,spm, translator, tags=['heading','p']):
  '''Strips xml into txt and translates it.
  '''

  xml_string = xml_string.replace('\\n','\n')
  xml_tree = BeautifulSoup(xml_string, features="xml")
  for tag in tags:
    instances = xml_tree.findAll(tag)
    for  ins in instances:
        text = ins.getText()
        ins.string = translate(text, tokenizer, spm, translator)

  return str(xml_tree).replace('\n','\\n')






# TRANSLATION FUNCTION (AFTER CALLING INIT FUNCITONS)

def translate_document(xml_text, src_lang, tokenizer,spm, translator):
    '''Function to call per each doc that should be translated'''

    xml_text = xml_text.decode('utf-8')
    stripped_text_xml = _normalize_input_string(xml_text.strip())
    translated_text_xml = translate_xml_tree(stripped_text_xml, tokenizer, spm, translator)

    return translated_text_xml




# INIT FUNCTIONS

def init_translator_model():
    '''Init model. Common for all langs
        Returns:            
            translator (model)
    
    '''
    

    try:
        translator = ctranslate2.Translator(TRANSLATION_MODEL_DIR, device='cuda')
    except:
        translator = ctranslate2.Translator(TRANSLATION_MODEL_DIR, device='cpu')


    return translator



def init_tokenizers(src_lang):
    '''Init tokenizer and pretokenizer 
        Returns:
            tokenizer
            spm
            trasnlator (model)
    
    '''
    

    tokenizer= TextTokenizer( _lang_code_translator(src_lang) )


    spm=AutoTokenizer.from_pretrained(TRANSLATION_MODEL_DIR_tok, src_lang=src_lang)
    return tokenizer, spm







## TRANSLATION CALL ORDER
#
# This is how this should be called

#translator_model = init_translator_model()
#
#for glg, eus, cat:
#    init_tokenizers(src_lang):
#
#
#for doc:
#    choose lang
#    translate_document(xml_text, src_lang, tokenizer,spm, translator):
