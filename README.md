# PDF2TXT PIPELINE - NextProcurement.

Dockerized pipeline to process pdf documents of public procurements.

![alt text](https://github.com/TeMU-BSC/NextProcurement_v2/blob/main/img/pdf2txt_scheme.png "Pipeline Scheme.")





## About the tool
The PDF2TXT module has been implemented as a dockerised pipeline, so that it can be easily used on any platform.
By providing a directory with PDF documents as input to it, output files with apache parquet extension are generated.

The text of each input PDF document is extracted (tables are omitted) and its language is identified.
The text is then automatically translated into the corresponding language in the ‘Translation Engine’ module, powered by artificial intelligence models.
The translated text is then converted into XML format (if desired). Finally, the information from the different documents is
saved in batches of parquet files, allowing easy export to other applications.






## Usage

Since it can be tricky to install the environment from scratch, a docker image is provided. 

### 1. Getting the Docker Image.

![alt text](https://github.com/TeMU-BSC/NextProcurement_v2/blob/main/img/docker_logo.jpg "docker icon.")



To use it there are to options:

 1. Download it directly from the docker repository, --> LINK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1.

 2. Creating it from the provided Dockerfile:

    
   To manually build the image (option 2) move to Docker/ folder and then type:

```bash
   docker build -t pdf2txt:2.0 .
```

### 2. Starting the container

We need now to run the docker container from the previous docker image. To do so we can do:

```bash
  docker run -it -it \
  -v <HOST LOCAL PATH>/input_folder/:/app/pdf2txt/<DOCKER_INPUT_PATH> \
  -v <HOST LOCAL PATH>/output_folder/:/app/pdf2txt/<DOCKER_OUTPUT_PATH> \
  pdf2txt:2.0
```
We need to bind local input/output path from our host to the docker image file system. This way, <HOST LOCAL PATH> will be a desired local path in your current machine and <DOCKER_INPUT_PATH>   and <DOCKER_OUTPUT_PATH> will be folders inside the docker virtual machine. Paths shoud be always be folders!


Keep in mind that once binded, files can be seen in the host without additional configuration.





Using the provided examples, a call to set up the docker container will be:

```bash
  docker run -it -it \
  -v <HOST LOCAL PATH>/pdf2txt/examples/input_folder/:/app/pdf2txt/examples/input_folder/ \
  -v <HOST LOCAL PATH>/pdf2txt/examples/output_folder/:/app/pdf2txt/examples/output_folder/ \
  pdf2txt:2.0
```



Once executed, a shell inside the docker virtual machine will be automatically provided.



### 3. Executing the pipeline
Now it is time to execute the pipeline, to do so just execute the following command:

```bash
  python pipeline/pipeline.py --input /app/pdf2txt/<DOCKER_INPUT_PATH> --output /app/pdf2txt/<DOCKER_OUTPUT_PATH> 
```

Or following the previous example
```bash
  python pipeline/pipeline.py --input /app/pdf2txt/examples/input_folder/ --output /app/pdf2txt/examples/output_folder/
```




### 4. (OPTIONAL) Switching provided content from XML to plain text.
The pipeline allows to extract pdf information in plain text instead of the deafult XML format, to do that just pass the --txt argument e.g.,

Or following the previous example
```bash
  python pipeline/pipeline.py --input /app/pdf2txt/examples/input_folder/ --output /app/pdf2txt/examples/output_folder/ --txt
```
If not, the output will be structured in sections as an XML document:

![alt text](https://github.com/TeMU-BSC/NextProcurement_v2/blob/main/img/page_0.jpg "Example of an page_X.jpg")






## PIPELINE'S OUTPUT

The output will be a folder containing different batches of apache parquet files. At most each parquet file will contain 100k documents.

Each parque file will contain the following columns:

- **procurement_id** : Procurement id or None if not possible to identify. Only for pdfs with names with format: ntp[0-9]*_*.pdf
- **original_doc_name** : Original document where the current row info. was extracted.
- **content** : XML/TXT content extracted from the pdf document.
- **lang** : 'es'/'ca'/'eu'/'gl', according to the content language.
- **translated_content** : If previous field was 'ca'/'eu'/'gl', the content translated to spanish. None if lang field was 'es'.



Here is and example of a parquet file:

![alt text](https://github.com/TeMU-BSC/NextProcurement_v2/blob/main/img/parquet.png "Example of an output.")


