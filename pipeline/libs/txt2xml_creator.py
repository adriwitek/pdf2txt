'''This scripts takes a .txt output from pdf2txt and parses it to a xml doc.



'''

import xml.etree.ElementTree as ET
import os




# New line XML char
#XML_NEWLINE_CHAR = '<br/>'
#XML_NEWLINE_CHAR = '\n'
XML_NEWLINE_CHAR = ' '





def process_doc(doc_txt):
    '''Takes a file doc (pdf2txt processed) and converts it to xml format

        Return xml whtin utf encoding.
    '''


    doc_xml = ET.Element('document')        
    current_section = None


    # Opened tags 
    opened_tag_heading = False 
    opened_tag_body_content = False
    new_paragraph_inside_content = False
    skip_empty_content = False




    for l in doc_txt.split('\n'):
        #print(l)
        ############################################
        # Handling pdf2txt marks
        if('## PAGE:') in l:
            #'''### PAGE:<NUMBER>##'''
            # Pages should not be important
            skip_empty_content = True
        elif('#HEADING:' in l):
            skip_empty_content = False
            opened_tag_heading = True
            opened_tag_body_content = False
            current_section = ET.Element('section')
            doc_xml.append (current_section) 
            heading = ET.SubElement(current_section, "heading") 
            #heading.text = l
        elif('#TABLE:' in l):
            continue
        elif('#DEFAULT:' in l):
            skip_empty_content = False
            opened_tag_heading = False
            new_paragraph_inside_content = True
            if(current_section is None): # in case there is not previous header mark
                current_section = ET.Element('section')
                doc_xml.append (current_section) 
        else:
            ############################################
            # Handling pdf2txt doc content
            ############################################
            if(skip_empty_content):
                continue
            # NO SECTION TAG HANDLING
            elif(current_section is None ):
                if(l == '\n'):
                    continue 
                # in case there is not previous mark (None of the possible ones)
                # This case should not be much common
                current_section = ET.Element('section')
                doc_xml.append (current_section) 
                opened_tag_body_content = True
                body = ET.SubElement(current_section, "body") 
                current_paragraph = ET.SubElement(body, "p") 
                # Adding content
                current_paragraph.text = l
            # HEDING HANDLING
            elif(opened_tag_heading):
                if(heading.text is None):
                    heading.text = l
                else:
                    heading.text += XML_NEWLINE_CHAR
                    heading.text += l
            #BODY CONTENT HANDLING
            elif(opened_tag_body_content): 
                # Adding content to alredy opened body tag
                if(new_paragraph_inside_content):
                    new_paragraph_inside_content = False
                    current_paragraph = ET.SubElement(body, "p") 
                # Handling content addition to what we alredy had
                if(current_paragraph.text is None):
                    current_paragraph.text = l
                else:
                    current_paragraph.text += XML_NEWLINE_CHAR
                    current_paragraph.text += l
            else: #Open body and current_paragraph
                opened_tag_heading = False
                opened_tag_body_content = True
                new_paragraph_inside_content = False
                body = ET.SubElement(current_section, "body") 
                current_paragraph = ET.SubElement(body, "p") 
                # Adding content
                if(current_paragraph.text is None):
                    current_paragraph.text = l
                else:
                    current_paragraph.text += XML_NEWLINE_CHAR
                    current_paragraph.text += l



    tree = ET.ElementTree(doc_xml) 
    tree = tree.getroot()

    return ET.tostring(tree, encoding='utf8')










#######################################
# EXAMPLE OF USE

#def  main():
#    
#        
#    process_doc('test/doc0.txt', 'test/doc0_converted.xml')
#    process_doc('test/doc1.txt', 'test/doc1_converted.xml')
#    process_doc('test/doc0.txt', None)


#if __name__ == "__main__":
#    main()






