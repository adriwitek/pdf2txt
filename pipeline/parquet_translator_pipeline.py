'''Additional tool to update already-generated parquets with only content column


    This will fill the traslation column in case language is not spanish to the
    corresponding lang (cat/eu/gl) --> es

    Content is assumed to be in plain txt



'''


import argparse
import logging
import sys
import os
import time
import pandas as pd
from fastparquet import ParquetFile
import translate as tr_model
from libs.lang_indentificator import *






def _lang_code_translator_2(src_lang):
    '''Aux func.
        INput: ca,gl or eu
    '''
   
    translated_lang_code = ''
    if( 'ca' in src_lang ):
      translated_lang_code = 'cat_Latn'
    elif( 'gl' in src_lang):
        translated_lang_code = 'glg_Latn'
    elif( 'eu' in src_lang):
        translated_lang_code = 'eus_Latn'
    else:
      raise Exception('Translation language couldnt be located.')
   


def translate_doc(doc_txt, lang, translator):
    '''Translate without lang_code already detected'''

    # Translate content
    translated_content = ''
    try:
        if (not 'es' in lang):
            translator_model = translator['model']
            if('ca' in lang):
                tokenizer, spm = translator['cat']
            elif('gl' in lang):
                tokenizer, spm = translator['glg']
            elif('eu' in lang):
                tokenizer, spm = translator['eus']
            equivalent_lang_code = _lang_code_translator_2(lang)
            translated_content = tr_model.translate(doc_txt, tokenizer, spm, translator_model)
    except Exception as e:
        logging.info(f'A translation could not be performed: Exception:\n {e}')

    return translated_content


def process_parquet_file(parquet_file, output_folder, translator):

    # Stats:
    logging.debug(f'Processing file:{parquet_file} ...')
    t_0 =   time.time() 

    # Open the parquet file and add new columns
    df = pd.read_parquet(parquet_file, engine='fastparquet')

    # Adding columns if they do not exist alredy
    if( ( 'alternative_lang' not in df) or ( 'lang' not in df)    ):
        logging.debug(f'Creating column: lang')
        df = df.assign(lang="")
        lang_code_included = False
    else:
        lang_code_included = True
        if('alternative_lang'  in df):
            try:
                df = df.rename(columns={'alternative_lang': 'lang'})
            except:
                logging.info('Unable to rename column alternative_lang to lang. Skipping...')
    if('translated_content' not in df):
        logging.debug(f'Creating column: translated_content')
        df = df.assign(translated_content="")



    # Iterating
    logging.debug(f'Performing translation of the parquet....')

    if(lang_code_included):
        for index, row in df.iterrows():
            logging.debug(f'Translating doc. with index {index}')
            lang =  row['lang'].decode("utf-8")
            doc_text =  row['content'].decode("utf-8")
            txt_translated =  translate_doc(doc_text,lang, translator)
            txt_translated_in_bytes =  bytes(txt_translated, 'utf-8')
            row['translated_content'] = txt_translated_in_bytes
            logging.debug(f'Translated doc. with index {index}')
    else:
        # not language code already detected
        for index, row in df.iterrows():
            logging.debug(f'Translating doc. with index {index}')
            doc_text =  row['content'].decode("utf-8")
            lang = get_language(doc_text)
            txt_translated =  translate_doc(doc_text,lang,translator)
            txt_translated_in_bytes =  bytes(txt_translated, 'utf-8')
            row['translated_content'] = txt_translated_in_bytes
    logging.debug(f'Translation task over the entire parquet filex executed!')


    # Rewriting
    logging.debug(f'Saving to disk...')
    parquet_filename = os.basename(parquet_file)
    final_name = str(parquet_filename).replace('.parq', '_with_translations.parq')
    final_path= os.path.join(output_folder, final_name)
    df.to_parquet(  final_path, 
                    #engine='fastparquet', 
                    engine='pyarrow', 
                    compression='lz4'
                  )
    logging.debug(f'File {final_path} fixed and saved!')


    # Stats Update
    t_1 = time.time() 
    time_spent =  (t_1 - t_0)/60.
    logging.info(f'Processing of:{parquet_file} FINISHED.\t Time spent:{time_spent:2f} min.')



def list_dir(directory):
    ''' Returns a list with parquet files in the given directory
    
    '''
    logging.info(f'Listing folder in  directory {directory} ...')
    dis_list = [ f'{directory}/{i}'  for i in  os.listdir(directory) if  i.endswith(".parq")]
    logging.info(f'DONE! Listed {len(dis_list)} parquet files.')

    return dis_list




def _parse_args():

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)


    parser.add_argument("-i", 
                        "--input",
                        default=None,
                        help="Input folder containing parquet files"
                        )
    parser.add_argument("-o", 
                        "--output",
                        help="Output path where parquets will be updated."
                        )
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        help='If passed, showing additional processing information.'
                        )
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)


    args = parser.parse_args()

    if(args.input is None):
        err_msg = "ERROR: --input argument must be provided!"
        raise  Exception( err_msg )
    elif( not os.path.isdir(args.input) ):
        err_msg = "ERROR: Provided input is not a directory:{args.input}"
        raise  Exception( err_msg )


    return args










def main(*args, **kwargs):

    args = _parse_args()
    if(args.verbose):
        logging.getLogger().setLevel(logging.DEBUG)


    list_parquets = list_dir(args.input)
    logging.debug(f'List of parquet files:\n{list_parquets}')


    # Loading translators
    logging.debug(f'Loading MT models...')
    translator_model = tr_model.init_translator_model()
    tokenizer_cat, spm_cat =  tr_model.init_tokenizers('cat_Latn')
    tokenizer_glg, spm_glg = tr_model.init_tokenizers('glg_Latn')
    tokenizer_eus, spm_eus =  tr_model.init_tokenizers('eus_Latn')
    translator = {}
    translator['model'] = translator_model
    translator['cat'] = ( tokenizer_cat, spm_cat )
    translator['glg'] = ( tokenizer_glg, spm_glg )
    translator['eus'] = ( tokenizer_eus, spm_eus )
    logging.debug(f'MT models looaded!')


    [process_parquet_file(i,args.output, translator) for i in list_parquets]


    logging.debug(f'Pipeline execution finished!')









if __name__ == '__main__':
    main()