from pdf_to_text import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import argparse
import sys
import os
import logging
import shutil


def process_doc(pdf_path, pipe, output_path, override = False):
    '''Processes and saves each pdf file'''

    # Create folder for output
    docname = os.path.basename(os.fsdecode(pdf_path)).split('.')[0]
    processed_output_folder= os.path.join(output_path,docname + '_processed')
    finished_processed_output_folder= os.path.join(output_path,docname + '_processed_finished')


    # Check doc is not already processed and if we want to override
    if  os.path.exists(finished_processed_output_folder):
        if(override):
            #logging.warning('Directory: ' + str(processed_output_folder) + ' already exits. Overwriting it!')
            shutil.rmtree(finished_processed_output_folder)           # Removes all the subdirectories!
        else:
            return 
        

    # Create temp folder
    if  os.path.exists(processed_output_folder):
        # If we are here, something went wrong so we overwrite files
        shutil.rmtree(processed_output_folder)           # Removes all the subdirectories!
    os.makedirs(processed_output_folder)


    image_and_paragraphs = get_paragraphs_from_pdf(pdf_path)

    with open(os.path.join(processed_output_folder, 'Doc_labeled_sections.txt'), 'w') as f:

        for page, (image, paragraphs) in enumerate(image_and_paragraphs):
            paragraphs_coords_and_labels = []
            f.write('## PAGE:'+ str(page) + '##\n\n')
            for text, coords, img in paragraphs:
                is_table = detect_table(img)
                if not is_table: #Process if not table
                    label = pipe(text)
                    f.write('#' + str(label[0]['label']).upper().strip() + ':\n' + text.rstrip() + '\n')
                else:
                    label = "Table"
                    f.write('#' + label.upper() + ':\n' + '...\n')

                paragraphs_coords_and_labels.append([coords, label])


            # img. output
            image_name = f"page_{page}.jpg"
            image_output_path =  os.path.join(processed_output_folder, image_name)
            get_results_visualization(image_output_path, image, paragraphs_coords_and_labels)


    # Rename folder in order to reflext the process is finished
    os.rename(processed_output_folder, finished_processed_output_folder)



# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')




def main(*args, **kwargs):


    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)



    parser.add_argument("-i", 
                        "--input",
                        default=None,
                        help="PDF doc or directory with several pdf documents"
                        )
    

    parser.add_argument('-l',
                        '--list_of_inputs', 
                        action='append', 
                        default=None,
                        help='List of docs to process. Each element must be preceded by the argument mark',
                        )

    
    parser.add_argument("-o", 
                        "--output",
                        help="Output path."
                        )


    parser.add_argument("-m", "--model", 
                        default="https://huggingface.co/BSC-LT/NextProcurement_pdfutils",
                        help="Path to used doc-processing model."
                        )
    
    parser.add_argument('--override_output',
                        action="store_true",
                        help='if set overrides the output'
                        )


    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)


    args = parser.parse_args()


    if(args.list_of_inputs is None and args.input is None):
        err_msg = "ERROR: --input or  --list_of_inputs argument must be provided!"
        raise  Exception( err_msg )


    if(args.output is None):
        err_msg = "ERROR: --output path must be provided!"
        raise  Exception( err_msg )

     
    # Load model and pipeline
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model))
    #pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device = -1)


    # List of N docs to process (helpul when we want to paralellize)
    if(args.list_of_inputs is not None):
        for pdf_path in args.list_of_inputs:
            filename = os.fsdecode(pdf_path)
            if filename.endswith(".pdf"): 
                try:
                    process_doc(os.path.join(pdf_path), pipe, args.output, args.override_output)
                except Exception as e:
                    err_msg = "ERROR: An error ocurred parsing the document called: " + filename + '. \n\t' + str(e)
                    #raise  Exception( err_msg)
                    print(err_msg)
            else:
                logging.info('Skipping non-pdf file: ' + filename)
    # An folder with docs
    elif(os.path.isdir(args.input) ):
        for pdf_path in os.listdir(args.input):
            filename = os.fsdecode(pdf_path)
            if filename.endswith(".pdf"): 
                try:
                    process_doc(os.path.join(args.input,pdf_path), pipe, args.output, args.override_output)
                except Exception as e:
                    err_msg = "ERROR: An error ocurred parsing the document called: " + filename + '. \n\t' + str(e)
                    #raise  Exception( err_msg)
                    print(err_msg)
            else:
                logging.info('Skipping non-pdf file: ' + filename)
       
    else:#single input doc
        process_doc(args.input, pipe, args.output, args.override_output)





  
    
    


if __name__ == "__main__":
    main()
