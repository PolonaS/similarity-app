import xml.etree.ElementTree
import shutil
import re
import os
from lib.tfpdf import tfidf, jaccard_similarity, tokenize, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from lib.db import create_table, insert


FILE = "pubmed_acl.xml"
CONTEXT_SIZE = 10
ABSTRACTS = 1000

OWN_COSINE_SIM = 1
SKLEARN_COSINE_SIM = 2
OWN_JACCARD_SIM = 3
SKLEARN_JACCARD_SIM = 4

root = xml.etree.ElementTree.parse(FILE).getroot()

count_abstracts = 0

db_abstracts = []
db_acronyms = []
db_unique_acronyms = []
db_full_forms = []
db_similarity = []


def parse_abstracts(el, results=None):
    if results is None:
        results = []
    for child in el:
        if child.tag == 'AbstractText':
            if child.text:
                results.append(child.text)
        parse_abstracts(child, results)
    return results


def parse():
    global db_abstracts, count_abstracts
    for article in root:
        cit = article.find("MedlineCitation")
        if cit:
            document_id = cit.find("PMID").text
            abstracts = parse_abstracts(article)
            if abstracts:
                inserted_abstracts = []
                for a in abstracts:
                    inserted_abstracts.append(a)
                    count_abstracts = count_abstracts + 1
                    if count_abstracts == ABSTRACTS:
                        create_orig_file(document_id, inserted_abstracts)
                        db_abstracts.append({'document_id': document_id, 'abstracts': inserted_abstracts})
                        return
                create_orig_file(document_id, inserted_abstracts)
                db_abstracts.append({'document_id': document_id, 'abstracts': inserted_abstracts})


def strip_acronym(text):
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(".", "")
    text = text.replace("!", "")
    text = text.replace(",", "")
    return text.strip()


def find_acronyms_in_string(text):
    result = []
    pattern = r'(\s\([A-Z]{2,}\)(?:\s|\,|\.|\!|\?))'
    for match in re.finditer(pattern, text):
        group = match.group()
        span = match.span()
        l_words = text[0:span[0]].strip().split(" ")
        r_words = text[span[1]:].strip().split(" ")
        left = " ".join(l_words[-CONTEXT_SIZE:])
        if left:
            left = left + " "
        right = " ".join(r_words[0:CONTEXT_SIZE])
        if right:
            right = " " + right
        context = left + group.strip() + right
        result.append({'original': group, 'striped': strip_acronym(group), 'span': span, 'context': context})
    return result


def find_acronyms():
    global db_abstracts, db_acronyms
    for abstracts in db_abstracts:
        document_id = abstracts['document_id']
        for abstract in abstracts['abstracts']:
            acronyms = find_acronyms_in_string(abstract)
            for acronym in acronyms:
                db_acronyms.append({'document_id': document_id, 'acronym': acronym})


def find_unique_acronyms():
    global db_unique_acronyms
    for acronyms in db_acronyms:
        striped_acronym = acronyms['acronym']['striped']
        if striped_acronym not in db_unique_acronyms:
            db_unique_acronyms.append(striped_acronym)


def find_full_forms_in_string(text, acronym):
    result = []
    words = text.split(" ")
    chars = list(acronym)
    c = 0
    for word in words:
        passed = False
        d = 0
        for char in chars:
            position = c + d
            if position < len(words):
                tested_word = words[position]
                if tested_word and char == tested_word[0]:
                    if d == len(chars) - 1:
                        passed = True
                        break
                    d = d + 1
                    continue
                else:
                    break
        if passed:
            span = [c, c + len(chars)]
            full_form = " ".join(words[span[0]:span[1]])
            l_words = words[0:span[0]]
            r_words = words[span[1]:]
            left = " ".join(l_words[-CONTEXT_SIZE:])
            if left:
                left = left + " "
            right = " ".join(r_words[0:CONTEXT_SIZE])
            if right:
                right = " " + right
            context = left + full_form + right
            result.append({'full_form': full_form, 'span': span, 'context': context})
        c = c + 1
    return result


def find_full_forms():
    global db_abstracts, db_full_forms, db_unique_acronyms
    for abstracts in db_abstracts:
        document_id = abstracts['document_id']
        for abstract in abstracts['abstracts']:
            for acronym in db_unique_acronyms:
                full_forms = find_full_forms_in_string(abstract, acronym)
                for full_form in full_forms:
                    db_full_forms.append({'document_id': document_id, 'acronym': acronym, 'full_form': full_form})


def calculate_similarity(text1, text2):
    tfidf_representation = tfidf([text1, text2])
    sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
                                    tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform([text1, text2]).toarray()
    return {
        OWN_COSINE_SIM: cosine_similarity(tfidf_representation[0], tfidf_representation[1]),
        SKLEARN_COSINE_SIM: cosine_similarity(sklearn_representation[0], sklearn_representation[1]),
        OWN_JACCARD_SIM: jaccard_similarity(tfidf_representation[0], tfidf_representation[1]),
        SKLEARN_JACCARD_SIM: jaccard_similarity(sklearn_representation[0], sklearn_representation[1])
    }


def find_similarity():
    global db_similarity
    for acronyms in db_acronyms:
        acronym_document_id = acronyms['document_id']
        acronym_striped = acronyms['acronym']['striped']
        acronym_context = acronyms['acronym']['context']
        for full_forms in db_full_forms:
            full_form_acronym = full_forms['acronym']
            if acronym_striped == full_form_acronym:
                full_form_document_id = full_forms['document_id']
                full_form = full_forms['full_form']['full_form']
                full_form_context = full_forms['full_form']['context']
                similarity = calculate_similarity(acronym_context, full_form_context)
                insert(acronym_document_id,
                       acronym_striped,
                       acronym_context,
                       full_form_document_id,
                       full_form,
                       full_form_context,
                       similarity[OWN_COSINE_SIM],
                       similarity[SKLEARN_COSINE_SIM],
                       similarity[OWN_JACCARD_SIM],
                       similarity[SKLEARN_JACCARD_SIM])


def create_results_folder():
    try:
        os.mkdir("results")
    except OSError:
        pass


def create_file(abstracts, file_name):
    f = open("results/" + file_name, "w")
    f.write("\n".join(abstracts))
    f.close()


def create_orig_file(id, abstracts):
    create_results_folder()
    create_file(abstracts, file_name=id + ".txt")


def main():
    shutil.rmtree("results", True)
    create_table()
    create_results_folder()
    parse()
    find_acronyms()
    find_unique_acronyms()
    find_full_forms()
    find_similarity()


if __name__ == "__main__":
    main()
