import sys
import time
import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ChromeOptions

# from selenium import webdriver
# page_num = 1
# base_url = f'https://www.propertyfinder.ae/en/search?c=2&fu=0&ob=mr&page={page_num}&rp=y'
PAGES = 200
property_finder_url = 'https://www.propertyfinder.ae/'


# chrome_options = Options()
# chrome_options.add_argument("--enable-javascript")

# # chrome_options.add_argument("--headless")

# driver = webdriver.Chrome(
#     ChromeDriverManager().install(), options=chrome_options)


def fetch_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:39.0)'}

    # driver.get(url)
    # time.sleep(2)
    # return driver.page_source.encode("utf-8")

    # session = HTMLSession()
    # request = session.get(
    #     url, headers=headers)

    # request.html.render()
    html = requests.get(url, headers=headers).text
    time.sleep(1.3)
    return html
    # return request.html.html


def get_soup_from_from_html(html):
    return BeautifulSoup(html, 'html.parser')


def get_all_card_list_divs(soup):
    return soup.find_all('div', attrs={'class': "card-list__item"})


def extact_info_from_div(card_header_div, card_content_div):

    apartment_location = card_header_div.contents[2].contents[1].text
    apartment_location = apartment_location.replace(",", "-")

    apartment_price = card_header_div.contents[0].contents[0].contents[0].text
    apartment_price = clean_price_text(apartment_price)

    # card content -> card info -> card info content -> card property amenties
    card_property_amneties = card_content_div.contents[1].contents[0].contents[0]

    apartment_bedrooms = get_div_content(
        "p", "card__property-amenity card__property-amenity--bedrooms", card_property_amneties)
    apartment_bathrooms = get_div_content(
        "p", "card__property-amenity.card__property-amenity--bathrooms", card_property_amneties)
    apartment_area = get_div_content(
        "p", "card__property-amenity card__property-amenity--area", card_property_amneties)
    listing_type = get_div_content(
        "p", "card__property-amenity.card__property-amenity--property-type", card_property_amneties)

    property_details = [str(listing_type), str(apartment_bedrooms), str(apartment_bathrooms),
                        str(apartment_area), str(apartment_price), str(apartment_location)]

    property_details = enclose_list_items_in_quotes(property_details)

    return property_details


def enclose_list_items_in_quotes(detail_list):
    enclosed_list = []
    for item in detail_list:
        enclosed_list.append(f'"{item}"')

    return enclosed_list

# removes '\n' and extra white spaces


def clean_price_text(apartment_price: str):
    apartment_price = remove_newlines(apartment_price)
    apartment_price = apartment_price.strip()
    apartment_price = " ".join(apartment_price.split())
    return apartment_price


def remove_newlines(text):
    return text.replace("\n", "\t")


def get_div_content(tag_type, class_text, parent_div):

    class_selector = ".".join(class_text.split(" "))
    try:
        div_text = parent_div.select(f"{tag_type}.{class_selector}")[0].text
    except IndexError:
        div_text = None

    return div_text


def open_file():
    data_file = open('apartment_data.csv', "a")
    return data_file


def add_headers_to_file():
    """
    adds headers to csv file.
    headers: 'listing_type', 'bedrooms', 'bathrooms', 'area', 'price', 'location', 'description'

    """
    data_file = open('apartment_data.csv', "w")
    data_file.write(
        "listing_type, bedrooms, bathrooms, area, price, location, description\n")
    data_file.close()


def close_file(file_to_close):
    file_to_close.close()


def get_property_description(listing_description_url):

    description_page_html = fetch_from_url(listing_description_url)
    description_page_soup = get_soup_from_from_html(
        description_page_html)
    description_div = description_page_soup.find(
        class_="property-page__description")

    # classes_in_description_tag = "text-trim property-description__text-trim text-trim--enabled text-trim--expanded"
    # class_list = classes_in_description_tag.split(" ")
    # text-trim property-description__text-trim text-trim--enabled text-trim--expanded
    property_description = get_div_content(
        "div", "text-trim property-description__text-trim", description_div)

    property_description = remove_newlines(property_description)
    property_description = clean_property_description_text(
        property_description)
    property_description = enclose_list_items_in_quotes([property_description])

    return property_description[0]


def clean_property_description_text(text):
    return text.replace(",", "-")


def main():

    data_file = open_file()
    # add_headers_to_file()

    for counter in range(1, 41):
        # dubai link:"https://www.propertyfinder.ae/en/search?c=2&fu=0&l=1&ob=mr&page=1&rp=y"
        base_url = f'https://www.propertyfinder.ae/en/search?c=2&fu=0&l=4&ob=mr&page={counter}&rp=y'

        html = fetch_from_url(base_url)
        soup = get_soup_from_from_html(html)

        card_count = 1

        # looping over each card list item
        for div in get_all_card_list_divs(soup):
            print(f"Page number: {counter}, card number: {card_count}")

            try:
                card_content = div.find_all('div', class_="card__content")[0]
            except IndexError:

                print("going to the next page")
                card_count += 1
                continue

            card_header = card_content.find(

                'div', class_="card__header")

            card_details = extact_info_from_div(card_header, card_content)
            data_file.write(",".join(card_details))

            # getting the description url
            sub_url = div.contents[0]['href']
            listing_description_url = property_finder_url + sub_url
            # -----------------------------

            # -----------------
            # description page

            property_description = get_property_description(
                listing_description_url)

            data_file.write("," + property_description + "\n")

            card_count += 1

    close_file(data_file)


if __name__ == "__main__":
    main()
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:39.0)'}
    # session = HTMLSession()
    # r = session.get(
    #     'https://www.propertyfinder.ae/en/search?c=2&fu=0&l=1&ob=mr&page=1&rp=y', headers=headers)
    # js = session.get(
    #     'https://www.propertyfinder.ae/dist/desktop/js/polyfills.a6d8614e23fcc6f90166.js').text
    # js2 = session.get(
    #     'https://www.propertyfinder.ae/dist/desktop/js/property-serp.78d53a778b78aca94367.js').text
    # r.html.render(script=[js, js2])

    # print(r.status_code)
    # print(r.html.html)
    # print(r.content)
