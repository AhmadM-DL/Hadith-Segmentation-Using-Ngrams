import requests
from bs4 import BeautifulSoup, Tag
import json
import os
import re
import pyarabic.araby as araby
import unicodedata as ud

rubbish_check = re.compile(r'[a-z1-9!@#$%^&*()<>?_+-=|"}{\][\]}]')


def extract_book(book_uri, output_path, verbose=1):
    book_page = requests.get(book_uri)
    book_soup = BeautifulSoup(book_page.content, 'html5lib')
    book_info = book_soup.find("div", class_="collection_info").find_all("div", class_="colindextitle")
    book_title = book_info[0].find_all("div")[1].text
    if len(book_info) == 2:
        book_description = book_info[1].text.strip()
    else:
        book_description = " No description available"
    book_number_of_volumes = int(book_soup.find_all("div", class_="book_number")[-1].text)

    extracted_book = {"Title": book_title,
                      "Description": book_description,
                      'Volumes': []}

    if verbose:
        print("The Book %s Contains %d Volumes:" % (book_title, book_number_of_volumes))

    for i in range(1, book_number_of_volumes + 1):

        uri = book_uri + "/" + str(i)

        if verbose:
            print("Getting Volume From %s" % uri)

        page = requests.get(uri)
        contents = page.content

        soup = BeautifulSoup(contents, 'html5lib')

        all_hadith = soup.find("div", class_="AllHadith")

        volume = {"Title": soup.find('div', class_='book_page_arabic_name arabic').text,
                  "Number": soup.find('div', class_='book_page_number').text.strip(),
                  "Chapters": []}

        # Default Chapter In case the volume doesn't have one or first hadith doesn't have one
        chapter = {"Number": str(0),
                   "Title": "باب مفترض",
                   "Hadiths": []}

        for child in all_hadith.children:

            if type(child) is Tag and child.has_attr("class"):

                tag_type = child.name
                tag_class = child.get("class")[0]

                if tag_type == "div" and tag_class == "chapter":

                    chapter = {"Number": child.find("div", class_="achapno").text[1:-1],
                               "Title": child.find("div", class_="arabicchapter arabic").text,
                               "Hadiths": []}

                    volume["Chapters"].append(chapter)
                    if verbose == 2:
                        print("Getting Chapter %s : %s" % (chapter["Number"], chapter["Title"]))

                elif tag_type == "div" and tag_class == "actualHadithContainer":

                    # Adding Default Chapter in case no one exist
                    if not volume["Chapters"]:
                        volume["Chapters"].append(chapter)
                        if verbose == 2:
                            print("Created Default Chapter %s : %s" % (chapter["Number"], chapter["Title"]))

                    sanads = child.findAll("span", class_="arabic_sanad arabic")

                    hadith = {"PreSanad": sanads[0].text,
                              "Body": child.find("span", class_="arabic_text_details arabic").text,
                              "PostSanad": sanads[1].text}
                    if verbose == 2:
                        print("Read Hadith")
                    volume["Chapters"][-1]["Hadiths"].append(hadith)

        extracted_book['Volumes'].append(volume)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    entire_book_file = open(output_path + extracted_book["Title"] + '.json', 'w')
    json.dump(extracted_book, entire_book_file, ensure_ascii=False)

    sanad, maten, atraf = book_maten_sanad_atraf_extractor(extracted_book)

    sanad_file = open(output_path + extracted_book["Title"] + '_sanad.txt', 'w')
    maten_file = open(output_path + extracted_book["Title"] + '_maten.txt', 'w')
    atraf_file = open(output_path + extracted_book["Title"] + '_atraf.txt', 'w')

    sanad_file.write(sanad)
    maten_file.write(maten)
    atraf_file.write(atraf)


def book_maten_sanad_atraf_extractor(book_dictionary, verbose=0):
    sanad_str = ""
    maten_str = ""
    atraf_str = ""

    for volume in book_dictionary['Volumes']:
        for chapter in volume["Chapters"]:
            for hadith in chapter["Hadiths"]:

                # Extract Sanad
                sanad_sentence = araby.strip_tashkeel(
                    hadith["PreSanad"].replace("\n", "").replace("\t", "").replace('\u200f', ''))

                sanad_sentence = ''.join(c for c in sanad_sentence if not ud.category(c).startswith('P'))
                sanad_sentence = ''.join(c for c in sanad_sentence if not rubbish_check.match(c))

                if verbose:
                    print("SANAD: " + sanad_sentence)

                sanad_str += sanad_sentence + "\n"

                # Extract Maten
                maten_sentence = araby.strip_tashkeel(
                    hadith["Body"].replace("\n", "").replace("\t", "").replace('\u200f', ''))

                maten_sentence = ''.join(c for c in maten_sentence if not ud.category(c).startswith('P'))
                maten_sentence = ''.join(c for c in maten_sentence if not rubbish_check.match(c))

                if verbose:
                    print("MATEN: " + maten_sentence)

                maten_str += maten_sentence + "\n"

                # Extract Atraf
                atraf_sentence = araby.strip_tashkeel(
                    hadith["PostSanad"].replace("\n", "").replace("\t", "").replace('\u200f', ''))

                atraf_sentence = ''.join(c for c in atraf_sentence if not ud.category(c).startswith('P'))
                atraf_sentence = ''.join(c for c in atraf_sentence if not rubbish_check.match(c))

                if verbose:
                    print("Atraf: " + atraf_sentence)

                atraf_str += atraf_sentence + "\n"

    return sanad_str, maten_str, atraf_str
