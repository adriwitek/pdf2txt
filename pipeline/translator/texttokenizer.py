#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
# Copyright (c) 2020 Jordi Mas i Hernandez <jmas@softcatala.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.

from __future__ import print_function
from translator.srx_segmenter import SrxSegmenter, parse
import os


def add_breakline_rule(rules,language):
    rules[language]["breaks"].append(["\n", #Before
                                      ""] # After
                                     )
    return rules


'''
    Splits text into sentences keeping spaces to allow later
    to reconstruct the same text but with translatabled text changed 
'''
class TextTokenizer:
    def __init__(self, language):
        srx_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segment.srx')
        self.rules = parse(srx_filepath)
        self.language = language 
        self.rules = add_breakline_rule(self.rules,language)
       
    def tokenize(self, sentence):
        strings = []
        translate = []

        segmenter = SrxSegmenter(self.rules[self.language], sentence)
        segments, whitespaces = segmenter.extract()

        for i in range(len(segments)):
            whitespace = whitespaces[i]
            if len(whitespace) > 0:
                strings.append(whitespace)
                translate.append(False)

            string = segments[i]
            strings.append(string)
            translate.append(True)

        return strings, translate

    def sentence_from_tokens(self, sentences, translate, translated):
        num_sentences = len(sentences)
        translation = ''
        for i in range(0, num_sentences):
            if translate[i] is True:
                translation += translated[i]
            else:
                translation += sentences[i]

        return translation.strip()