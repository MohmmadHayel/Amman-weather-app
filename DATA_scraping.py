

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import csv
import time

csv_path = "C:/Users/User/Documents/Projects/DE_Final_project/scrabed_data.csv"
all_info = []

service_obj = Service(ChromeDriverManager().install())
browser = webdriver.Chrome(service=service_obj)

def get_data(current_url):
    browser.get(current_url)
    time.sleep(5) 
    
    dropdown_element = browser.find_element("id", "wt-his-select")
    dropdown = Select(dropdown_element)
    
    days_values = []
    for option in dropdown.options:
        val = option.get_attribute("value")
        days_values.append(val)

    for val in days_values:
        print(f"جاري سحب: {val}")
        dropdown.select_by_value(val)
        time.sleep(2) 
        
        soup = BeautifulSoup(browser.page_source, "lxml")
        table = soup.find("table", {"id": "wt-his"})
        
        if table:
            rows = table.find("tbody").find_all("tr")
            for row in rows:
                time_text = row.find("th").text.strip()
                cells = row.find_all("td")
                
                if len(cells) > 7: # تأكد أن الصف يحتوي على كل البيانات
                    all_info.append({
                        "time": time_text,
                        "temperature": cells[1].text.strip(),
                        "status": cells[2].text.strip(),
                        "wind speed": cells[3].text.strip(),
                        "direction": cells[4].text.strip(),
                        "Humidity": cells[5].text.strip(),
                        "Barometer": cells[6].text.strip(),
                        "Visibility": cells[7].text.strip(),
                        "date": val
                    })

def save_to_file():
    if all_info:
        keys = all_info[0].keys()
        with open(csv_path, "w", newline='', encoding='utf-8') as scrabed_data:
            dict_writer = csv.DictWriter(scrabed_data, keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_info)
            print(f"Done! File created with {len(all_info)} rows.")

try:
    for i in range(1, 13):
        target_url = f"https://www.timeanddate.com/weather/jordan/amman/historic?month={i}&year=2025"
        print(f"--- بدء سحب شهر رقم {i} ---")
        get_data(target_url)
    
    save_to_file()

finally:
    browser.quit()