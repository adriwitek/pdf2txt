# NextProcurement Project (v2)

Pipeline (version 2) for NextProcurement Project.


## About the tool
It labels each pdf section according its function in the document (Title, header, paragraph...)

## Usage

### Downloading AI model

Due to github size limitations this model should be downloaded from https://huggingface.co/BSC-LT/NextProcurement_pdfutils . To achieve that we can do:


```bash
cd pipeline/models/

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/BSC-LT/NextProcurement_pdfutils
```


### Environment Setup
It's recommended to create a new enviroment in order to avoid lib. version errors.The tool was tested under **python 3.7**, I highly recoomed using the same. Another option is to use the provided docker described in the next section.

YOu can create a virtual env with the following command:

```bash
python -m venv env 
```

and activating it with:


```bash
source env/bin/activate
```

Then, dependencies should be installed:



#### SYSTEM DEPENDENCIES:
 - poppler (this one is a little bit hard to install, this might helps you: https://cbrunet.net/python-poppler/installation.html)
 - leptonica
 - terrasec-ocr*
   - *This files are alse needed by tesseact. Download and locate them under /usr/share/tesseract-ocr/4.00/tessdata/ (Depending on the configuration it can also be under /usr/local/share/tessdata or /usr/share/tessdata):
     - Spanish OCR Model: https://github.com/tesseract-ocr/tessdata/blob/main/spa.traineddata
     - Catalan OCR Model: https://github.com/tesseract-ocr/tessdata/blob/main/cat.traineddata
     - Euskera OCR Model: https://github.com/tesseract-ocr/tessdata/blob/main/eus.traineddata
     - Galician OCR Model: https://github.com/tesseract-ocr/tessdata/blob/main/glg.traineddata



#### PYTHON DEPENDENCIES:
```bash
pip install --upgrade pip
pip install --use-pep517 -r requirements.txt
```



### Docker

Since it can be tricky to install the environment from scratch, a docker image is provided. 

To use it there are to options:

 1. Download it directly from the docker repository, available [here](https://hub.docker.com/r/adriwitek/pdf2txt).

 2. Creating it from the provided Dockerfile.

    
   To manually build the image (option 2) move to Docker/ folder and then type:

```bash
   docker build -t pdf2txt:1.0 .
```

And then:

```bash
   docker run -it pdf2txt:1.0
```

To run the image creating a container. An interactive shell will be given with the "-it" argument.


### Usage
Firstly remember to have the virtual enviroment already activated. Then,

```bash
python pipeline/pipeline.py  \
        --input <pdf file> | <folder with pdf files>> \
        --output <output folder> \
        --model <model paht> [NOT REQUIERED, will take default model if not specified] (desired model classify the sections (excluding tables) (title, header,etc.) )
```

#### Example of use

```bash
python pipeline/pipeline.py  \
        --input examples_input_folder \
        --output  output_folder_nextprocurement_project \
        --model pipeline/models/nextprocurement_pdfutils/ \
```
#### Ouput format

*output* will be a folder containing a subfolder for each document in *input*. Each subfolder will contain:
- **Doc_labeled_sections.txt** : A plain .txt:  with a label above each paragraph
- **page_X.jpg** : Several .jpg files for each page in the document. Each image has colored rectangle over each paragraph and its corresponding label. An example looks like this:

![alt text](https://github.com/TeMU-BSC/NextProcurement_v2/blob/main/img/page_0.jpg "Example of an page_X.jpg")




