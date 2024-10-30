'''Pipeline to process a given folder with pdfs

    It recieves a folder containing pdfs and produces parquet files with processed information.

'''


import argparse
import sys
import os
import logging
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


from libs.pdf_to_text import *
from libs.txt2xml_creator import *
from libs.lang_indentificator import *


# MACROS 
DEVICE = -1


# NOT CONFIGURABLE MACROS
#PDF2TXT_CLASSIFIER_MODEL = 'https://huggingface.co/BSC-LT/NextProcurement_pdfutils'
PDF2TXT_CLASSIFIER_MODEL = '/app/pdf2txt/pipeline/models/nextprocurement_pdfutils'
MAX_TOKENIZER_LENGTH=512



def list_dir(directory):
    ''' Returns a list with pdf files in the given directory
    
    '''

    logging.info(f'Listing folder in  directory {directory} ...')
    dis_list = [ f'{directory}/{i}'  for i in  os.listdir(directory) if  i.endswith(".pdf")]
    logging.info(f'DONE! Listed {len(dis_list)} pdfs.')

    return dis_list





def _get_txt_from_pdf_aux(pdf_path, pipe):
    '''Processes and saves each pdf file'''

    all_paragraphs = get_paragraphs_from_pdf(os.path.join(pdf_path))
    for page_number,  paragraphs_in_page in enumerate(all_paragraphs):
        buffer_txt = []
        #f.write('## PAGE:'+ str(page_number) + '##\n\n')
        buffer_txt.append('## PAGE:'+ str(page_number) + '##\n\n')
        for text, coords, is_table in paragraphs_in_page:
            if not is_table: #Process if not table
                label = pipe((text[MAX_TOKENIZER_LENGTH]) if len(text) > MAX_TOKENIZER_LENGTH else text)
                #f.write('#' + str(label[0]['label']).upper().strip() + ':\n' + text.rstrip() + '\n')
                buffer_txt.append('#' + str(label[0]['label']).upper().strip() + ':\n' + text.rstrip() + '\n')
            else:
                label = "Table"
                #f.write('#' + label.upper() + ':\n' + '...\n')
                buffer_txt.append('#' + label.upper() + ':\n' + '...\n')
    # Return txt
    return ''.join(buffer_txt)

  


def get_txt_from_pdf (pdf_path, pipe):
    '''Func that _get_txt_from_pdf_aux()  handles unexpected errors
         so the comprehension list does not fail its execution.
    
        Returns:
            txt of the pdf, None if error occurred.
    '''

    filename = os.fsdecode(pdf_path)
    if filename.endswith(".pdf"): 
        try:
          txt =  _get_txt_from_pdf_aux(pdf_path, pipe)
          return txt
        except Exception as e:
            err_msg = "ERROR: An error ocurred parsing the document: " + filename + '. \n\t' + str(e)
            #raise  Exception( err_msg)
            print(err_msg)
            return None
    else:
        logging.info('Skipping non-pdf file: ' + filename)
        return None






def process_pdf(pdf_path, pipe):
    '''Func that process each pdf and return the desired output'''


    # Txt processing
    doc_clean_txt = get_txt_from_pdf (pdf_path, pipe)
    if(doc_clean_txt) is None: # Skipping doc
        return

    doc_xml_txt = process_doc(doc_clean_txt)

    # Lang detection
    lang = get_language(doc_clean_txt)






    # Debuggg
    print(f'New doc! (lang:{lang})')
    print(f'-------------------TEXT:------')
    print(doc_clean_txt)
    print(f'-------------------XML:------')
    print(doc_xml_txt)




# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')




def _parse_args():


    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)



    parser.add_argument("-i", 
                        "--input",
                        default=None,
                        help="Input folder containing pdf documents"
                        )
    

    #parser.add_argument('-l',
    #                    '--list_of_inputs_and_outputs', 
    #                    action='append', 
    #                    default=None,
    #                    help='List of docs to process. Each element must be preceded by the argument mark. Each pair input,output must be separated by commas.',
    #                    )

    
    parser.add_argument("-o", 
                        "--output",
                        help="Output path where parquets will be generated."
                        )


    #parser.add_argument("-m", "--model", 
    #                    default="https://huggingface.co/BSC-LT/NextProcurement_pdfutils",
    #                    help="Path to used doc-processing model."
    #                    )
    #

    parser.add_argument('--override_output',
                        action="store_true",
                        help='if set overrides the output'
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

    if(args.output is None ):
        err_msg = "ERROR: --output path must be provided!"
        raise  Exception( err_msg )

    return args


def main(*args, **kwargs):


    
    args = _parse_args()
     


    # Load model and pipeline
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(str(PDF2TXT_CLASSIFIER_MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(str(PDF2TXT_CLASSIFIER_MODEL))
    #pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device = DEVICE)



    list_of_pdfs = list_dir(args.input)

    

    # Process list of pdfs
    [ process_pdf(pdf_path, pipe) for pdf_path in list_of_pdfs]


    ##### TODO I SHOULD HANDLE IT SO IT GIVES EVERYTHING I NEEEEEEEEED







  
    
    


if __name__ == "__main__":
    main()



'''Run example:


    # With a list of files:

        > python3 pipeline/pipeline.py     --output /gpfs/scratch/bsc88/bsc88621/NextProcurement/_ttest_output_several_files  --model /gpfs/projects/bsc88/NextProcurement/Pipeline/pipeline/pdf2txt/model_section_classificator -l /gpfs/scratch/bsc88/MN4/bsc88/bsc88621/NextProcurement/datantp01067774_Pliego_Prescripciones_tecnicas_URI.pdf -l /gpfs/scratch/bsc88/MN4/bsc88/bsc88621/NextProcurement/datantp01024989_Documento_Publicado_URI:5.pdf 


'''