'''Pipeline to process a given folder with pdfs

    It recieves a folder containing pdfs and produces parquet files with processed information.

'''


import argparse
import sys
import os
import logging
import shutil
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


from libs.pdf_to_text import *
from libs.txt2xml_creator import *
from libs.lang_indentificator import *
#from libs.machine_translation.texttokenizer import TextTokenizer
#import translate as tr_model
import translator.translator as tr_model
import pymupdf  # PyMuPDFv
import torch



###################################
# CONFIGURABLE MACROS 
###################################


#DEVICE = -1
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU



###################################
# NOT CONFIGURABLE MACROS
###################################

# Classifier Model
#PDF2TXT_CLASSIFIER_MODEL = 'https://huggingface.co/BSC-LT/NextProcurement_pdfutils'
PDF2TXT_CLASSIFIER_MODEL = '/app/pdf2txt/pipeline/models/nextprocurement_pdfutils'
MAX_TOKENIZER_LENGTH=512
MT_TRANSLATOR_PATH='/app/pdf2txt/pipeline/models/salamandra_engine/salamandraTA-2B'
MT_TRANSLATOR_PATH_GGUF = 'salamandraTA-2B.Q2_K.gguf'

#NTP_PATTERN = r"ntp\d+_.*\.pdf"
NTP_PATTERN = r"PL\d+_.*\.pdf"








def _lang_code_translator_2(src_lang):
    '''
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
   

    return translated_lang_code



def list_dir(directory):
    ''' Returns a list with pdf files in the given directory
    
    '''
    logging.info(f'Listing folder in  directory {directory} ...')
    dis_list = [ f'{directory}/{i}'  for i in  os.listdir(directory) if  i.endswith(".pdf")]
    logging.info(f'DONE! Listed {len(dis_list)} pdfs.')

    return dis_list




def _get_txt_from_pdf_aux_without_marks(pdf_path):
    '''Processes and saves each pdf file'''

    all_paragraphs = get_paragraphs_from_pdf(os.path.join(pdf_path))
    buffer_txt = []
    for page_number,  paragraphs_in_page in enumerate(all_paragraphs):
        for text, coords, is_table in paragraphs_in_page:
            if not is_table: #Process if not table
                buffer_txt.append(text.rstrip() + '\n')
    # Return txt
    return ''.join(buffer_txt)




def _get_txt_from_pdf_aux(pdf_path, pipe):
    '''Processes and saves each pdf file'''

    all_paragraphs = get_paragraphs_from_pdf(os.path.join(pdf_path))
    buffer_txt = []
    for page_number,  paragraphs_in_page in enumerate(all_paragraphs):
        buffer_txt.append('## PAGE:'+ str(page_number) + '##\n\n')
        for text, coords, is_table in paragraphs_in_page:
            if not is_table: #Process if not table
                label = pipe((text[MAX_TOKENIZER_LENGTH]) if len(text) > MAX_TOKENIZER_LENGTH else text)
                #f.write('#' + str(label[0]['label']).upper().strip() + ':\n' + text.rstrip() + '\n')
                buffer_txt.append('#' + str(label[0]['label']).upper().strip() + ':\n' + text.rstrip() + '\n')
            else:
                label = "Table"
                buffer_txt.append('#' + label.upper() + ':\n' + '...\n')
    # Return txt
    return ''.join(buffer_txt)


def is_pdf_text_based(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            if page.get_text():  # If any page contains text, it's a text PDF
                return True
        return False
    except Exception as e:
        err = f'Error detecting pdf with pymupdf.\n\t-->File:{pdf_path} \n\tException:{e}'
        return False

def extract_pdf_text(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)  # Open the PDF
        text = "\n".join(page.get_text() for page in doc)  # Extract text from each page
        return text
    except Exception as e:
        err = f'Error processing pdf with pymupdf.\n\t-->File:{pdf_path} \n\tException:{e}'
        raise err
        return None


def get_txt_from_pdf(pdf_path, pipe, save_output_as_txt,args):
    '''Func that _get_txt_from_pdf_aux()  handles unexpected errors
         so the comprehension list does not fail its execution.
    
        Returns:
            txt of the pdf, None if error occurred.
    '''

    filename = os.fsdecode(pdf_path)
    if filename.endswith(".pdf"): 
        try:
          content = None
          if(save_output_as_txt):# txt

            if is_pdf_text_based(pdf_path):
                # Written pdfs
                if(args.process_only_scanned_pdfs):
                    pass
                    print(f'Skipping writen-pdf type since arg process_only_scanned_pdfs was provided: {pdf_path}')
                else:
                    # fastpymupdf
                    content = extract_pdf_text(pdf_path)
            else:
                # Scanned pdfs
                if(args.process_only_written_pdfs):
                    pass
                    print(f'Skipping scanned-pdf type since arg process_only_written_pdfs was provided: {pdf_path}')
                else:
                    # ocrtesseract
                    #content =  _get_txt_from_pdf_aux_without_marks(pdf_path, pipe)
                    content =  _get_txt_from_pdf_aux_without_marks(pdf_path)



          else: # xml
            content =  _get_txt_from_pdf_aux(pdf_path, pipe)
          return content
        
        except Exception as e:
            err_msg = "ERROR: An error ocurred parsing the document: " + filename + '. \n\t' + str(e)
            #raise  Exception( err_msg)
            print(err_msg)
            return None
    else:
        logging.info('Skipping non-pdf file: ' + filename)
        return None



def _extract_proc_ntp_id(pdf_name):
    '''Tries to extract ntpXXX id from pdf name
        Returns:
            nptXXX string-id, None if not possible to find
    '''


    if (re.match(NTP_PATTERN, pdf_name) is not None ):
        try:
            id= pdf_name.split('_')[0]
            return id
        except:
            return None
    else: 
        return None


def process_pdf(pdf_path, pipe, translator, save_output_as_txt,args, pfds_names_to_exclude = []):
    '''Func that process each pdf and return the desired output'''



   # Pdf basename
    original_pdf_name = os.path.basename(pdf_path)


    # Skipping already processed option
    if( (args.exclude_this_docs is not None) and  (len(pfds_names_to_exclude) > 0)  ):
        # Check if pdf processing will be skipped
        if(original_pdf_name in pfds_names_to_exclude):
            logging.info(f'--> Skipping pdf document as demanded by argument option. PDF filename: {original_pdf_name}')
            return None


    # Proc ID  (if can be located)
    ntp_id = _extract_proc_ntp_id(original_pdf_name)


    # Txt processing
    doc_clean_txt = get_txt_from_pdf(pdf_path, pipe, save_output_as_txt,args) # (txt or "txt to xml marks" depending or argument)
    if(doc_clean_txt) is None: # Skipping doc
        return None
    
    # Lang detection
    lang = get_language(doc_clean_txt)

    # XML parsing
    if(save_output_as_txt):
        original_content = doc_clean_txt
    else:
        original_content = process_doc(doc_clean_txt)


    if(args.dont_translate_docs):
        # do not translate
        translated_content = ''
    else:
        # Translate content
        translated_content = ''
        try:
            model = translator['model']       
            tokenizer = translator['tokenizer']
            translated_content = tr_model.translate_doc(original_pdf_name,lang, doc_clean_txt,  tokenizer, model)
        except Exception as e:
            print(f'Translation could not be done: Exception:\n {e}')



    return (ntp_id, original_pdf_name , original_content , lang, translated_content)






# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')




def chunk_list_in_n_slices(l, n):
    ''' Primitive
        n number of elements per chunck
    '''
    for i in range(0, len(l), n):  
        yield l[i:i + n] 




def _create_parquet_file(slice_list_of_procurements, list_index, pipe, translator, output_folder, save_output_as_txt,args, pfds_names_to_exclude = []):
    '''
        list_index: just to keep track of the general slice index
        translator: dict with translator models
    '''




    columns = {
        'procurement_id': 'string',
        'original_doc_name' : 'string',
        'content': 'string',  # This column will contain long text
        'lang': 'string',
        'translated_content': 'string',
    }



    # Get results of each procuremetn
    #info = [process_pdf(pdt_path, pipe, translator, save_output_as_txt,args) for pdt_path in slice_list_of_procurements]
    info = [result for pdt_path in slice_list_of_procurements if (result := process_pdf(pdt_path, pipe, translator, save_output_as_txt, args, pfds_names_to_exclude)) is not None]

    
    if len(info) == 0:
        logging.info(f'Quality check. Avoiding to write empy parquet chunck...')
        return 
    else:
        # Parquet generation
        df_content = [ {'procurement_id': ntp_id, 
                        'original_doc_name': original_pdf_name, 
                        'content': doc_xml_txt ,
                        'lang': lang ,
                        'translated_content': tranlated_doc_xml_txt,
                        } for (ntp_id, original_pdf_name , doc_xml_txt , lang, tranlated_doc_xml_txt) in  info 
                    ]
        df = pd.DataFrame(df_content)
        n_of_docs_in_slice = df.shape[0]
    
    
    
        # Writing parquet
        os.makedirs(output_folder,exist_ok = True)
        output_file_name = f'procurements_file_{list_index}_containing_{n_of_docs_in_slice}_docs.parq'
        output_path = os.path.join(output_folder,output_file_name)
        logging.info(f'Creating parquet with index {list_index} as file named as:{output_path} ...')
    
        #write(PARQUET_OUTPUT_PATH, df) # C MEMORY ERROR
        df.to_parquet(  output_path, 
                        #engine='fastparquet', 
                        engine='pyarrow', 
                        compression='lz4'
                      )
        logging.info(f'File Saved!')

 




def read_txt_with_list_of_pdfs(path):
    list_excluded_pdfs = []
    try:
        logging.debug(f'Reading excluding pdf list in the txt file:{path} ...')
        with open(path, 'r') as file:
            list_excluded_pdfs = [line.strip().split('/')[-1] for line in file if  line.strip().endswith('.pdf')] #pdfnames
        logging.debug(f'Done!')

    except Exception as e:
        msg = f'Error, ther was an error parsing the path: {path}. Err:{e}'
        return list_excluded_pdfs

    return list_excluded_pdfs




def _parse_args():

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)


    parser.add_argument("-i", 
                        "--input",
                        default=None,
                        help="Input folder containing pdf documents"
                        )
    parser.add_argument("-o", 
                        "--output",
                        help="Output path where parquets will be generated."
                        )
    parser.add_argument('--txt',
                        action="store_true",
                        help='If passed, content will be saved inside parquet files as plain txt instead of default structured XML format. '
                        )
    parser.add_argument( "--max_docs_per_parquet",
                        help="Number of max. docs that will be contained in each parquet file (inside the output folder.)",
                        default=100000,
                        type=int
                        )
    parser.add_argument('--process_only_scanned_pdfs',
                        action="store_true",
                        help='If passed, only scanned pdfs (slow ones) will be processed, if not, all of them will be considered. '
                        )
    parser.add_argument('--process_only_written_pdfs',
                        action="store_true",
                        help='If passed, only pdfs with selectable text (fast ones) will be processed, if not, all of them will be considered. '
                        )
    parser.add_argument('--dont_translate_docs',
                        action="store_true",
                        help='If passed, documents wont be translated.'
                        )
    
    parser.add_argument('--exclude_this_docs',
                        required=False,
                        type=str,
                        default=None,
                        help='List to a txt file. Each line will contain a path to a pdf doc. This list of files will be skipped. (Useful when they are alredy processed) Comparison criteria will be the pdf filename.'
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

    if(args.process_only_scanned_pdfs is True and args.process_only_written_pdfs is True):
        print('ERROR: You cannot specify both process_only_scanned_pdfs and process_only_written_pdfs arguments.')




    # Pre-load user info
    print(f'INFO: Each parquet file will contain a max. of {int(args.max_docs_per_parquet)} pdf processed documents.')
    ## XML parsing
    if(args.txt):
        print('--> Output content will be plain txt.')
    else:
        print('--> Output content will be plain XML structured data.')




    # Classifier: Loading model and pipeline
    if(args.txt):
        pipe = None
    else: 
        # This is only needed to tag xml sections.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(str(PDF2TXT_CLASSIFIER_MODEL))
        model = AutoModelForSequenceClassification.from_pretrained(str(PDF2TXT_CLASSIFIER_MODEL))
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device = DEVICE)




    # Tranlation: Load model(s) and pipeline
    if(not args.dont_translate_docs):
        # Do not load models if they are not going to be translated
        translator_model,tokenizer = tr_model.init_models(  model_id = MT_TRANSLATOR_PATH,
                                                            gguf_model_name = MT_TRANSLATOR_PATH_GGUF ,
        )
        translator = {}
        translator['model'] = translator_model
        translator['tokenizer'] = tokenizer
    else:           
        translator = {}
        translator['model'] = None
        translator['tokenizer'] = None



    # Slicing procurements for different parquet files slices
    list_of_pdfs = list_dir(args.input)
    if(len(list_of_pdfs) <1):
        logging.info('No pdf documents fount in provided input! Quitting...')
        return 
    list_of_lists = chunk_list_in_n_slices(list_of_pdfs, int(args.max_docs_per_parquet))


    if(args.exclude_this_docs is not None):
        logging.info(f'Reading list of files to skip ...')
        pfds_names_to_exclude= read_txt_with_list_of_pdfs(args.exclude_this_docs)
        logging.debug(f'The following pdf names will be skipped if found:\n{pfds_names_to_exclude}')
        logging.info(f'Done!')






    cpl_processor = [(list_index, sublist_of_pdfs) for list_index, sublist_of_pdfs in enumerate(list_of_lists)]
    if(args.exclude_this_docs is None):
        [ _create_parquet_file(sublist_of_pdfs, list_index, pipe, translator, args.output, args.txt,args) for list_index, sublist_of_pdfs in cpl_processor ]
    else:
        [ _create_parquet_file(sublist_of_pdfs, list_index, pipe, translator, args.output, args.txt,args, pfds_names_to_exclude) for list_index, sublist_of_pdfs in cpl_processor ]





if __name__ == "__main__":
    main()


